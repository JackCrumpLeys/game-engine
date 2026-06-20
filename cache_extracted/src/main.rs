#![feature(duration_constructors_lite)]
use std::{
    error::Error,
    time::{Duration, UNIX_EPOCH},
};

use game_engine::GameApp;

fn main() -> Result<(), Box<dyn Error>> {
    // Configure logger at runtime
    fern::Dispatch::new()
        // Perform allocation-free log formatting
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{} {} {}] {}",
                get_time(),
                record.level(),
                record.target(),
                message
            ))
        })
        // Add blanket level filter -
        .level(log::LevelFilter::Debug)
        .chain(std::io::stdout())
        // Apply globally
        .apply()?;

    let mut app = GameApp::default();

    app.run()?;

    Ok(())
}

fn get_time() -> String {
    let time = std::time::SystemTime::now()
        .checked_add(Duration::from_hours(13)) // NZST offset
        .unwrap()
        .duration_since(UNIX_EPOCH)
        .unwrap();
    let secs = time.as_secs();
    let mins = secs / 60;
    let hours = mins / 60;

    let mut days = hours / 24;

    let mut year = 1970;

    // 1. Determine the current year
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days >= days_in_year {
            days -= days_in_year;
            year += 1;
        } else {
            break;
        }
    }

    // `days` now holds the number of full days passed in the current year (0-indexed)
    let day_of_year = days + 1; // Convert to 1-indexed for month calculation

    // 2. An array of the number of days in each month (for a non-leap year)
    let month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 0;
    let mut temp_days = day_of_year;

    // 3. Determine the current month
    for (i, &days_in_month) in month_lengths.iter().enumerate() {
        let mut current_month_length = days_in_month;
        // Adjust for February in a leap year
        if i == 1 && is_leap(year) {
            current_month_length = 29;
        }

        if temp_days <= current_month_length {
            month = i as u64 + 1; // Month is 1-12
            break;
        } else {
            temp_days -= current_month_length;
        }
    }

    format!(
        "{:02}-{:02}-{:04} {:02}:{:02}:{:02}",
        temp_days,
        month,
        year,
        hours % 24,
        mins % 60,
        secs % 60
    )
}

fn is_leap(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}
