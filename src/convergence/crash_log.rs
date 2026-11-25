//! Persistent crash & runtime diagnostics logging for Symbiote: Abiogenesis.
//!
//! Installs a global panic hook that records panic message, file/line location,
//! thread name, and full Rust backtrace to `logs/crash_<timestamp>.log` and
//! appends lifecycle events to `logs/runtime.log`.

use std::backtrace::Backtrace;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const LOG_DIR: &str = "logs";
pub const RUNTIME_LOG: &str = "logs/runtime.log";

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn install_panic_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let ts = unix_secs();
        let thread = std::thread::current()
            .name()
            .unwrap_or("unnamed")
            .to_string();
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "unknown".to_string());
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "non-string panic payload".to_string()
        };
        let bt = Backtrace::force_capture();

        let report = format!(
            "=== SYMBIOTE: ABIOGENESIS CRASH REPORT ===\n\
             Timestamp (Unix): {ts}\n\
             Version: {}\n\
             Thread: {thread}\n\
             Location: {location}\n\
             Message: {payload}\n\n\
             Backtrace:\n{bt}\n",
            env!("CARGO_PKG_VERSION")
        );

        let _ = fs::create_dir_all(LOG_DIR);
        let crash_path = Path::new(LOG_DIR).join(format!("crash_{ts}.log"));
        let _ = fs::write(&crash_path, &report);
        log_event(&format!(
            "FATAL PANIC at {location}: {payload} (saved {})",
            crash_path.display()
        ));

        prev(info);
    }));
}

pub fn log_event(message: &str) {
    let _ = fs::create_dir_all(LOG_DIR);
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(RUNTIME_LOG)
    {
        let _ = writeln!(file, "[{}] {}", unix_secs(), message);
    }
}
