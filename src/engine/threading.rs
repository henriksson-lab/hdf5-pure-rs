use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{
    AtomicBool, AtomicI32, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering,
};
use std::sync::{Arc, Barrier, Condvar, Mutex, Once};
use std::thread::{self, JoinHandle, ThreadId};

use crate::error::ErrorStack;
use crate::hl::context::ApiContext;

#[derive(Debug, Default)]
pub struct ThreadPackage {
    initialized: AtomicBool,
}

impl ThreadPackage {
    pub fn init_package(&self) {
        self.initialized.store(true, Ordering::SeqCst);
    }

    pub fn term_package(&self) {
        self.initialized.store(false, Ordering::SeqCst);
    }

    pub fn pthread_first_thread_init(&self) {
        self.init_package();
    }

    pub fn c11_first_thread_init(&self) {
        self.init_package();
    }

    pub fn win32_process_enter(&self) {
        self.init_package();
    }

    pub fn top_term_package(&self) {
        self.term_package();
    }

    pub fn tinfo_term(&self) {
        self.term_package();
    }
}

#[derive(Debug, Default)]
pub struct TsMutex {
    state: Mutex<TsMutexState>,
}

#[derive(Debug, Default)]
struct TsMutexState {
    owner: Option<ThreadId>,
}

impl TsMutex {
    pub fn mutex_init() -> Self {
        Self::default()
    }

    pub fn mutex_trylock(&self) -> bool {
        let current = thread::current().id();
        if let Ok(mut state) = self.state.try_lock() {
            if state.owner.is_none() {
                state.owner = Some(current);
                return true;
            }
        }
        false
    }

    pub fn mutex_lock(&self) {
        while !self.mutex_trylock() {
            thread::yield_now();
        }
    }

    pub fn mutex_unlock(&self) {
        let current = thread::current().id();
        let mut state = self.state.lock().expect("mutex poisoned");
        match state.owner {
            Some(owner) if owner == current => state.owner = None,
            Some(_) => panic!("mutex unlocked from a non-owning thread"),
            None => panic!("mutex unlocked while not locked"),
        }
    }

    pub fn mutex_destroy(self) {}

    pub fn mutex_acquire(&self) {
        self.mutex_lock();
    }

    pub fn mutex_release(&self) {
        self.mutex_unlock();
    }

    pub fn api_lock(&self) {
        self.mutex_lock();
    }

    pub fn api_mutex_release(&self) {
        self.mutex_unlock();
    }

    pub fn api_unlock(&self) {
        self.mutex_unlock();
    }
}

#[derive(Debug)]
pub struct TsSemaphore {
    count: Mutex<usize>,
    cv: Condvar,
}

impl TsSemaphore {
    pub fn semaphore_init(count: usize) -> Self {
        Self {
            count: Mutex::new(count),
            cv: Condvar::new(),
        }
    }

    pub fn semaphore_wait(&self) {
        let mut count = self.count.lock().expect("semaphore poisoned");
        while *count == 0 {
            count = self.cv.wait(count).expect("semaphore poisoned");
        }
        *count -= 1;
    }

    pub fn semaphore_signal(&self) {
        let mut count = self.count.lock().expect("semaphore poisoned");
        *count += 1;
        self.cv.notify_one();
    }

    pub fn semaphore_destroy(self) {}
}

pub struct TsThread {
    handle: Option<JoinHandle<()>>,
}

impl TsThread {
    pub fn thread_create<F>(task: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self {
            handle: Some(thread::spawn(task)),
        }
    }

    pub fn thread_join(mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    pub fn thread_detach(mut self) {
        self.handle.take();
    }

    pub fn thread_yield() {
        thread::yield_now();
    }
}

#[derive(Debug)]
pub struct TsCondvar {
    cv: Condvar,
}

impl TsCondvar {
    pub fn cond_init() -> Self {
        Self { cv: Condvar::new() }
    }

    pub fn cond_signal(&self) {
        self.cv.notify_one();
    }

    pub fn cond_broadcast(&self) {
        self.cv.notify_all();
    }

    pub fn cond_destroy(self) {}
}

#[derive(Debug, Clone)]
pub struct TsBarrier {
    inner: Arc<Barrier>,
}

impl TsBarrier {
    pub fn barrier_init(participants: usize) -> Self {
        Self {
            inner: Arc::new(Barrier::new(participants)),
        }
    }

    pub fn barrier_wait(&self) -> bool {
        self.inner.wait().is_leader()
    }

    pub fn barrier_destroy(self) {}
}

#[derive(Debug, Default)]
pub struct TsRwLock {
    state: Mutex<RwLockState>,
    cv: Condvar,
    stats: RwLockStats,
}

#[derive(Debug, Default)]
struct RwLockState {
    readers: HashMap<ThreadId, usize>,
    writer: Option<ThreadId>,
    writer_recursion: usize,
}

#[derive(Debug, Default, Clone)]
pub struct RwLockStats {
    pub rdlocks: Arc<AtomicU64>,
    pub wrlocks: Arc<AtomicU64>,
    pub rdunlocks: Arc<AtomicU64>,
    pub wrunlocks: Arc<AtomicU64>,
}

impl TsRwLock {
    pub fn rwlock_init() -> Self {
        Self::default()
    }

    pub fn rwlock_rdlock(&self) {
        self.rdlock(false);
    }

    fn rdlock(&self, recursive: bool) {
        let current = thread::current().id();
        let mut state = self.state.lock().expect("rwlock poisoned");
        while state.writer.is_some() && !(recursive && state.writer == Some(current)) {
            state = self.cv.wait(state).expect("rwlock poisoned");
        }
        *state.readers.entry(current).or_insert(0) += 1;
        self.update_stats_rdlock();
    }

    pub fn rwlock_rdunlock(&self) {
        self.rdunlock();
    }

    fn rdunlock(&self) {
        let current = thread::current().id();
        let mut state = self.state.lock().expect("rwlock poisoned");
        let count = state
            .readers
            .get_mut(&current)
            .expect("rwlock read-unlocked by a thread without a read lock");
        *count -= 1;
        if *count == 0 {
            state.readers.remove(&current);
        }
        if state.readers.is_empty() {
            self.cv.notify_all();
        }
        self.update_stats_rd_unlock();
    }

    pub fn rwlock_wrlock(&self) {
        self.wrlock(false);
    }

    fn wrlock(&self, recursive: bool) {
        let current = thread::current().id();
        let mut state = self.state.lock().expect("rwlock poisoned");
        if recursive && state.writer == Some(current) {
            state.writer_recursion += 1;
            self.update_stats_wr_lock();
            return;
        }
        while state.writer.is_some() || !state.readers.is_empty() {
            state = self.cv.wait(state).expect("rwlock poisoned");
        }
        state.writer = Some(current);
        state.writer_recursion = 1;
        self.update_stats_wr_lock();
    }

    pub fn rwlock_trywrlock(&self) -> bool {
        self.trywrlock(false)
    }

    fn trywrlock(&self, recursive: bool) -> bool {
        let current = thread::current().id();
        if let Ok(mut state) = self.state.try_lock() {
            if recursive && state.writer == Some(current) {
                state.writer_recursion += 1;
                self.update_stats_wr_lock();
                return true;
            }
            if state.writer.is_none() && state.readers.is_empty() {
                state.writer = Some(current);
                state.writer_recursion = 1;
                return true;
            }
        }
        false
    }

    pub fn rwlock_wrunlock(&self) {
        self.wrunlock();
    }

    fn wrunlock(&self) {
        let current = thread::current().id();
        let mut state = self.state.lock().expect("rwlock poisoned");
        match state.writer {
            Some(owner) if owner == current => {
                state.writer_recursion = state.writer_recursion.saturating_sub(1);
                if state.writer_recursion == 0 {
                    state.writer = None;
                    self.cv.notify_all();
                }
            }
            Some(_) => panic!("rwlock write-unlocked from a non-owning thread"),
            None => panic!("rwlock write-unlocked while not write-locked"),
        }
        self.update_stats_wr_unlock();
    }

    pub fn rwlock_destroy(self) {}

    pub fn rec_rwlock_init() -> Self {
        Self::rwlock_init()
    }

    pub fn rec_rwlock_destroy(self) {}

    pub fn rec_rwlock_rdlock(&self) {
        self.rdlock(true);
    }

    pub fn rec_rwlock_wrlock(&self) {
        self.wrlock(true);
    }

    pub fn rec_rwlock_rdunlock(&self) {
        self.rdunlock();
    }

    pub fn rec_rwlock_wrunlock(&self) {
        self.wrunlock();
    }

    pub fn update_stats_rdlock(&self) {
        self.stats.rdlocks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn update_stats_rd_lock_delay(&self) {
        self.update_stats_rdlock();
    }

    pub fn update_stats_rd_unlock(&self) {
        self.stats.rdunlocks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn update_stats_wr_lock(&self) {
        self.stats.wrlocks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn update_stats_wr_lock_delay(&self) {
        self.update_stats_wr_lock();
    }

    pub fn update_stats_wr_unlock(&self) {
        self.stats.wrunlocks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn rec_rwlock_get_stats(&self) -> (u64, u64, u64, u64) {
        (
            self.stats.rdlocks.load(Ordering::Relaxed),
            self.stats.wrlocks.load(Ordering::Relaxed),
            self.stats.rdunlocks.load(Ordering::Relaxed),
            self.stats.wrunlocks.load(Ordering::Relaxed),
        )
    }

    pub fn rec_rwlock_reset_stats(&self) {
        self.stats.rdlocks.store(0, Ordering::Relaxed);
        self.stats.wrlocks.store(0, Ordering::Relaxed);
        self.stats.rdunlocks.store(0, Ordering::Relaxed);
        self.stats.wrunlocks.store(0, Ordering::Relaxed);
    }

    pub fn rec_rwlock_print_stats_fmt<W>(&self, out: &mut W) -> fmt::Result
    where
        W: fmt::Write + ?Sized,
    {
        let (rd, wr, rdu, wru) = self.rec_rwlock_get_stats();
        write!(
            out,
            "rdlock={rd}, wrlock={wr}, rdunlock={rdu}, wrunlock={wru}"
        )
    }

    pub fn rec_rwlock_print_stats_into(&self, out: &mut String) {
        out.clear();
        self.rec_rwlock_print_stats_fmt(out)
            .expect("writing to String cannot fail");
    }
}

pub struct TsThreadPool {
    tasks: Mutex<Vec<JoinHandle<()>>>,
}

impl Default for TsThreadPool {
    fn default() -> Self {
        Self {
            tasks: Mutex::new(Vec::new()),
        }
    }
}

impl TsThreadPool {
    pub fn pool_add_task<F>(&self, task: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.tasks
            .lock()
            .expect("pool poisoned")
            .push(thread::spawn(task));
    }

    pub fn pool_do(&self) {
        let mut tasks = self.tasks.lock().expect("pool poisoned");
        for task in tasks.drain(..) {
            let _ = task.join();
        }
    }

    pub fn pool_free(self) {
        self.pool_do();
    }

    pub fn pool_destroy(self) {
        self.pool_do();
    }
}

#[derive(Debug, Default)]
pub struct ThreadInfo {
    api_ctx: ApiContext,
    err_stack: ErrorStack,
    dlftt: usize,
}

impl ThreadInfo {
    pub fn tinfo_init() -> Self {
        Self::default()
    }

    pub fn tinfo_create() -> Self {
        Self::default()
    }

    pub fn thread_id() -> ThreadId {
        thread::current().id()
    }

    pub fn get_api_ctx_ptr(&mut self) -> &mut ApiContext {
        &mut self.api_ctx
    }

    pub fn get_err_stack(&mut self) -> &mut ErrorStack {
        &mut self.err_stack
    }

    pub fn get_dlftt(&self) -> usize {
        self.dlftt
    }

    pub fn set_dlftt(&mut self, value: usize) {
        self.dlftt = value;
    }

    pub fn inc_dlftt(&mut self) -> usize {
        self.dlftt += 1;
        self.dlftt
    }

    pub fn dec_dlftt(&mut self) -> usize {
        self.dlftt = self.dlftt.saturating_sub(1);
        self.dlftt
    }

    pub fn tinfo_destroy(self) {}
}

#[derive(Debug)]
pub struct TsKey<T> {
    value: Mutex<Option<T>>,
}

impl<T> TsKey<T> {
    pub fn key_create() -> Self {
        Self {
            value: Mutex::new(None),
        }
    }

    pub fn key_set_value(&self, value: T) {
        *self.value.lock().expect("key poisoned") = Some(value);
    }

    pub fn key_with_value<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        let value = self.value.lock().expect("key poisoned");
        value.as_ref().map(f)
    }

    pub fn key_take_value(&self) -> Option<T> {
        self.value.lock().expect("key poisoned").take()
    }

    pub fn key_delete(self) {}
}

impl<T: Clone> TsKey<T> {
    #[deprecated(
        since = "0.1.0",
        note = "use key_with_value for borrowed access or key_take_value for ownership transfer"
    )]
    pub fn key_get_value(&self) -> Option<T> {
        self.key_get_value_cloned()
    }

    pub fn key_get_value_cloned(&self) -> Option<T> {
        self.value.lock().expect("key poisoned").clone()
    }
}

pub fn once(once: &Once, f: impl FnOnce()) {
    once.call_once(f);
}

pub fn atomic_init_int(value: i32) -> AtomicI32 {
    AtomicI32::new(value)
}

pub fn atomic_load_int(value: &AtomicI32) -> i32 {
    value.load(Ordering::SeqCst)
}

pub fn atomic_store_int(value: &AtomicI32, new_value: i32) {
    value.store(new_value, Ordering::SeqCst);
}

pub fn atomic_fetch_add_int(value: &AtomicI32, addend: i32) -> i32 {
    value.fetch_add(addend, Ordering::SeqCst)
}

pub fn atomic_fetch_sub_int(value: &AtomicI32, subtrahend: i32) -> i32 {
    value.fetch_sub(subtrahend, Ordering::SeqCst)
}

pub fn atomic_destroy_int(_value: AtomicI32) {}

pub fn atomic_init_uint(value: u32) -> AtomicU32 {
    AtomicU32::new(value)
}

pub fn atomic_load_uint(value: &AtomicU32) -> u32 {
    value.load(Ordering::SeqCst)
}

pub fn atomic_store_uint(value: &AtomicU32, new_value: u32) {
    value.store(new_value, Ordering::SeqCst);
}

pub fn atomic_fetch_add_uint(value: &AtomicU32, addend: u32) -> u32 {
    value.fetch_add(addend, Ordering::SeqCst)
}

pub fn atomic_fetch_sub_uint(value: &AtomicU32, subtrahend: u32) -> u32 {
    value.fetch_sub(subtrahend, Ordering::SeqCst)
}

pub fn atomic_destroy_uint(_value: AtomicU32) {}

pub fn atomic_init_voidp<T>(value: *mut T) -> AtomicPtr<T> {
    AtomicPtr::new(value)
}

pub fn atomic_exchange_voidp<T>(value: &AtomicPtr<T>, new_value: *mut T) -> *mut T {
    value.swap(new_value, Ordering::SeqCst)
}

pub fn atomic_destroy_voidp<T>(_value: AtomicPtr<T>) {}

pub fn atomic_init_usize(value: usize) -> AtomicUsize {
    AtomicUsize::new(value)
}

#[cfg(test)]
mod tests {
    use super::{
        atomic_fetch_add_int, atomic_init_int, atomic_load_int, TsKey, TsMutex, TsRwLock,
        TsThreadPool,
    };

    #[test]
    fn threading_mutex_and_pool_do_real_work() {
        let mutex = TsMutex::mutex_init();
        assert!(mutex.mutex_trylock());
        mutex.mutex_unlock();

        let value = std::sync::Arc::new(atomic_init_int(0));
        let pool = TsThreadPool::default();
        for _ in 0..4 {
            let value = value.clone();
            pool.pool_add_task(move || {
                atomic_fetch_add_int(&value, 1);
            });
        }
        pool.pool_do();
        assert_eq!(atomic_load_int(&value), 4);
    }

    #[test]
    fn threading_key_supports_borrowed_and_owned_access() {
        #[derive(Debug, PartialEq, Eq)]
        struct NoClone(String);

        let key = TsKey::key_create();
        key.key_set_value(NoClone("value".to_string()));

        assert_eq!(
            key.key_with_value(|value| value.0.as_str() == "value"),
            Some(true)
        );
        assert_eq!(key.key_take_value(), Some(NoClone("value".to_string())));
        assert!(key.key_with_value(|_| ()).is_none());
    }

    #[test]
    fn recursive_rwlock_stats_into_replaces_stale_output() {
        let lock = TsRwLock::rec_rwlock_init();
        lock.rec_rwlock_rdlock();
        lock.rec_rwlock_rdunlock();
        lock.rec_rwlock_wrlock();
        lock.rec_rwlock_wrunlock();

        let mut stats = String::from("stale");
        lock.rec_rwlock_print_stats_into(&mut stats);
        assert_eq!(stats, "rdlock=1, wrlock=1, rdunlock=1, wrunlock=1");

        lock.rec_rwlock_reset_stats();
        lock.rec_rwlock_print_stats_into(&mut stats);
        assert_eq!(stats, "rdlock=0, wrlock=0, rdunlock=0, wrunlock=0");
    }

    #[test]
    fn recursive_rwlock_allows_same_thread_write_reentry() {
        let lock = TsRwLock::rec_rwlock_init();
        lock.rec_rwlock_wrlock();
        lock.rec_rwlock_wrlock();
        assert!(!lock.rwlock_trywrlock());
        lock.rec_rwlock_wrunlock();
        assert!(!lock.rwlock_trywrlock());
        lock.rec_rwlock_wrunlock();
        assert!(lock.rwlock_trywrlock());
        lock.rwlock_wrunlock();
    }

    #[test]
    fn rwlock_holds_read_and_write_state_until_unlock() {
        let lock = TsRwLock::rwlock_init();

        lock.rwlock_rdlock();
        assert!(!lock.rwlock_trywrlock());
        lock.rwlock_rdunlock();

        assert!(lock.rwlock_trywrlock());
        assert!(!lock.rwlock_trywrlock());
        lock.rwlock_wrunlock();

        assert!(lock.rwlock_trywrlock());
        lock.rwlock_wrunlock();
    }

    #[test]
    #[should_panic(expected = "mutex unlocked while not locked")]
    fn mutex_unlock_without_lock_panics() {
        TsMutex::mutex_init().mutex_unlock();
    }

    #[test]
    #[should_panic(expected = "rwlock read-unlocked by a thread without a read lock")]
    fn rwlock_read_unlock_without_lock_panics() {
        TsRwLock::rwlock_init().rwlock_rdunlock();
    }

    #[test]
    #[should_panic(expected = "rwlock write-unlocked while not write-locked")]
    fn rwlock_write_unlock_without_lock_panics() {
        TsRwLock::rwlock_init().rwlock_wrunlock();
    }
}
