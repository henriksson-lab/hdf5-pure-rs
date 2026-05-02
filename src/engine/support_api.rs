#![allow(dead_code, non_snake_case)]

use std::cmp::Ordering;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Condvar, Mutex, Once};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::error::{Error, Result};

static LIBRARY_OPEN: AtomicBool = AtomicBool::new(false);
static LIBRARY_TERMINATING: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone)]
pub struct H5Timer {
    pub started: Option<Instant>,
    pub elapsed: Duration,
}

impl Default for H5Timer {
    fn default() -> Self {
        Self {
            started: None,
            elapsed: Duration::ZERO,
        }
    }
}

#[derive(Debug, Default)]
pub struct H5TsMutex {
    locked: Mutex<bool>,
}

#[derive(Debug, Default)]
pub struct H5TsCond {
    cond: Condvar,
}

#[derive(Debug, Default)]
pub struct H5TsSemaphore {
    permits: Mutex<usize>,
    cond: Condvar,
}

#[derive(Debug, Default)]
pub struct H5TsRwLock {
    state: Mutex<(usize, bool)>,
    cond: Condvar,
}

fn unsupported_support(name: &str) -> Error {
    Error::Unsupported(format!(
        "{name} requires platform/MPI behavior not implemented in pure-Rust mode"
    ))
}

pub fn H5_ITER_ERROR() -> i32 {
    -1
}

pub fn H5_BEFORE_USER_CB() {}

pub fn H5_BEFORE_USER_CB_NOERR() {}

pub fn H5_mpi_comm_dup() -> Result<()> {
    Err(unsupported_support("H5_mpi_comm_dup"))
}

pub fn H5_mpi_comm_free() -> Result<()> {
    Err(unsupported_support("H5_mpi_comm_free"))
}

pub fn H5_mpi_comm_cmp() -> Result<Ordering> {
    Err(unsupported_support("H5_mpi_comm_cmp"))
}

pub fn H5_mpio_gatherv_alloc() -> Result<()> {
    Err(unsupported_support("H5_mpio_gatherv_alloc"))
}

pub fn H5_mpio_gatherv_alloc_simple() -> Result<()> {
    Err(unsupported_support("H5_mpio_gatherv_alloc_simple"))
}

pub fn H5_mpio_get_file_sync_required() -> bool {
    false
}

pub fn H5A__open_common() -> Result<()> {
    Err(Error::Unsupported(
        "attribute open-common duplicate is tracked in the attribute API".into(),
    ))
}

pub fn H5_gmtime_r(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn H5_localtime_r(time: SystemTime) -> u64 {
    H5_gmtime_r(time)
}

pub fn H5T__set_precision() -> Result<()> {
    Err(Error::Unsupported(
        "datatype precision duplicate is tracked in the datatype API".into(),
    ))
}

pub fn H5_buffer_dump(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn H5TS_api_lock() {}

pub fn H5_bandwidth(bytes: u64, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs == 0.0 {
        0.0
    } else {
        bytes as f64 / secs
    }
}

pub fn H5_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn H5_now_usec() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros()
}

pub fn H5_get_time() -> SystemTime {
    SystemTime::now()
}

pub fn H5__timer_get_timevals() -> (u64, u128) {
    (H5_now(), H5_now_usec())
}

pub fn H5_timer_init() -> H5Timer {
    H5Timer::default()
}

pub fn H5_timer_start(timer: &mut H5Timer) {
    timer.started = Some(Instant::now());
}

pub fn H5_timer_stop(timer: &mut H5Timer) -> Duration {
    if let Some(started) = timer.started.take() {
        timer.elapsed = timer.elapsed.saturating_add(started.elapsed());
    }
    timer.elapsed
}

pub fn H5_timer_get_times(timer: &H5Timer) -> Duration {
    timer.elapsed
}

pub fn H5_timer_get_total_times(timer: &H5Timer) -> Duration {
    timer.elapsed
}

pub fn H5_timer_get_time_string(duration: Duration) -> String {
    format!("{:.6}s", duration.as_secs_f64())
}

pub fn H5__init_package() {
    LIBRARY_OPEN.store(true, AtomicOrdering::SeqCst);
}

pub fn H5_init_library() {
    H5__init_package();
}

pub fn H5_term_library() {
    LIBRARY_TERMINATING.store(true, AtomicOrdering::SeqCst);
    LIBRARY_OPEN.store(false, AtomicOrdering::SeqCst);
}

pub fn H5dont_atexit() {}

pub fn H5garbage_collect() {}

pub fn H5__debug_mask(mask: u64) -> u64 {
    mask
}

pub fn H5__mpi_delete_cb() -> Result<()> {
    Err(unsupported_support("H5__mpi_delete_cb"))
}

pub fn H5get_libversion() -> (u32, u32, u32) {
    (0, 1, 0)
}

pub fn H5_check_version(major: u32, minor: u32, release: u32) -> bool {
    H5get_libversion() == (major, minor, release)
}

pub fn H5check_version(major: u32, minor: u32, release: u32) -> bool {
    H5_check_version(major, minor, release)
}

pub fn H5open() {
    H5_init_library();
}

pub fn H5atclose() {}

pub fn H5close() {
    H5_term_library();
}

pub fn H5allocate_memory(size: usize) -> Vec<u8> {
    vec![0; size]
}

pub fn H5resize_memory(mut bytes: Vec<u8>, size: usize) -> Vec<u8> {
    bytes.resize(size, 0);
    bytes
}

pub fn H5free_memory(_bytes: Vec<u8>) {}

pub fn H5is_library_threadsafe() -> bool {
    true
}

pub fn H5is_library_terminating() -> bool {
    LIBRARY_TERMINATING.load(AtomicOrdering::SeqCst)
}

pub fn H5_user_cb_prepare() {}

pub fn H5_user_cb_restore() {}

pub fn H5_HAVE_PARALLEL() -> bool {
    false
}

pub fn H5__checksum_crc_update(seed: u32, bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .fold(seed, |acc, byte| acc.rotate_left(5) ^ u32::from(*byte))
}

pub fn H5_checksum_crc(bytes: &[u8]) -> u32 {
    H5__checksum_crc_update(0, bytes)
}

pub fn H5UC_create() -> usize {
    1
}

pub fn H5UC_decr(count: &mut usize) -> usize {
    *count = count.saturating_sub(1);
    *count
}

pub fn H5TS_mutex_init() -> H5TsMutex {
    H5TsMutex::default()
}

pub fn H5TS_mutex_trylock(mutex: &H5TsMutex) -> bool {
    if let Ok(mut locked) = mutex.locked.try_lock() {
        if !*locked {
            *locked = true;
            return true;
        }
    }
    false
}

pub fn H5TS_mutex_destroy(_mutex: H5TsMutex) {}

pub fn H5TS_semaphore_init(permits: usize) -> H5TsSemaphore {
    H5TsSemaphore {
        permits: Mutex::new(permits),
        cond: Condvar::new(),
    }
}

pub fn H5TS_semaphore_destroy(_sem: H5TsSemaphore) {}

pub fn H5TS_once(once: &Once, f: fn()) {
    once.call_once(f);
}

pub fn H5TS_cond_init() -> H5TsCond {
    H5TsCond::default()
}

pub fn H5TS_cond_destroy(_cond: H5TsCond) {}

pub fn H5TS_semaphore_signal(sem: &H5TsSemaphore) {
    if let Ok(mut permits) = sem.permits.lock() {
        *permits = permits.saturating_add(1);
        sem.cond.notify_one();
    }
}

pub fn H5TS_semaphore_wait(sem: &H5TsSemaphore) {
    if let Ok(mut permits) = sem.permits.lock() {
        while *permits == 0 {
            permits = sem.cond.wait(permits).expect("semaphore wait poisoned");
        }
        *permits = permits.saturating_sub(1);
    }
}

pub fn H5TS_rwlock_init() -> H5TsRwLock {
    H5TsRwLock::default()
}

pub fn H5TS_rwlock_destroy(_lock: H5TsRwLock) {}

pub fn H5TS_rwlock_rdlock(lock: &H5TsRwLock) {
    let mut state = lock.state.lock().expect("rwlock poisoned");
    while state.1 {
        state = lock.cond.wait(state).expect("rwlock wait poisoned");
    }
    state.0 = state.0.saturating_add(1);
}

pub fn H5TS_rwlock_rdunlock(lock: &H5TsRwLock) {
    if let Ok(mut state) = lock.state.lock() {
        state.0 = state.0.saturating_sub(1);
        if state.0 == 0 {
            lock.cond.notify_all();
        }
    }
}

pub fn H5TS_rwlock_wrlock(lock: &H5TsRwLock) {
    let mut state = lock.state.lock().expect("rwlock poisoned");
    while state.1 || state.0 != 0 {
        state = lock.cond.wait(state).expect("rwlock wait poisoned");
    }
    state.1 = true;
}

pub fn H5TS_rwlock_trywrlock(lock: &H5TsRwLock) -> bool {
    if let Ok(mut state) = lock.state.try_lock() {
        if !state.1 && state.0 == 0 {
            state.1 = true;
            return true;
        }
    }
    false
}

pub fn H5TS_rwlock_wrunlock(lock: &H5TsRwLock) {
    if let Ok(mut state) = lock.state.lock() {
        state.1 = false;
        lock.cond.notify_all();
    }
}

pub fn H5TS_key_create() -> usize {
    0
}

pub fn H5TS_key_delete(_key: usize) {}

pub fn H5TS_key_set_value<T>(_key: usize, value: T) -> T {
    value
}

pub fn H5TS_key_get_value(_key: usize) -> Option<()> {
    None
}

pub fn H5_trace_args_bool(value: bool) -> String {
    value.to_string()
}

pub fn H5_trace_args_cset(value: u8) -> String {
    value.to_string()
}

pub fn H5_trace_args_close_degree(value: u8) -> String {
    value.to_string()
}

pub fn H5_trace_args(args: &[String]) -> String {
    args.join(", ")
}

pub fn H5_trace(message: &str) -> String {
    message.to_string()
}

pub fn H5TS_thread_create<F>(f: F) -> JoinHandle<()>
where
    F: FnOnce() + Send + 'static,
{
    thread::spawn(f)
}

pub fn H5TS_thread_join(handle: JoinHandle<()>) -> Result<()> {
    handle
        .join()
        .map_err(|_| Error::InvalidFormat("thread panicked".into()))
}

pub fn H5TS_thread_detach(_handle: JoinHandle<()>) {}

pub fn H5TS_thread_yield() {
    thread::yield_now();
}

pub fn H5FD__copy_plist() -> Result<()> {
    Err(Error::Unsupported(
        "VFD property-list copy duplicate is tracked in the VFD API".into(),
    ))
}

pub fn H5_subfiling_dump_iovecs() -> Result<()> {
    Err(unsupported_support("H5_subfiling_dump_iovecs"))
}

pub fn H5_make_time(secs: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(secs)
}

pub fn H5_get_localtime_str(time: SystemTime) -> String {
    H5_gmtime_r(time).to_string()
}

pub fn H5_get_win32_times() -> Result<()> {
    Err(unsupported_support("H5_get_win32_times"))
}

pub fn H5_get_utf16_str(bytes: &[u16]) -> Result<String> {
    String::from_utf16(bytes).map_err(|_| Error::InvalidFormat("invalid UTF-16 string".into()))
}

pub fn H5_build_extpath(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}/{name}")
    }
}

pub fn H5_nanosleep(nanos: u64) {
    thread::sleep(Duration::from_nanos(nanos));
}

pub fn H5_expand_windows_env_vars(value: &str) -> String {
    value.to_string()
}

pub fn H5_strndup(value: &str, len: usize) -> String {
    value.chars().take(len).collect()
}

pub fn H5_dirname(value: &str) -> String {
    Path::new(value)
        .parent()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".to_string())
}

pub fn H5_basename(value: &str) -> String {
    Path::new(value)
        .file_name()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|| value.to_string())
}

pub fn H5_get_option<T: Copy>(value: Option<T>, default: T) -> T {
    value.unwrap_or(default)
}

pub fn H5_strcasestr(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .to_ascii_lowercase()
        .find(&needle.to_ascii_lowercase())
}

pub fn is_host_little_endian() -> bool {
    cfg!(target_endian = "little")
}

pub fn H5EA__dblock_create() -> Result<()> {
    Err(Error::Unsupported(
        "extensible-array data block creation is not implemented".into(),
    ))
}
