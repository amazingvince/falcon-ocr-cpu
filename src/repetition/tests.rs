//! Tests of where the stop ends exact and counting loops, and of the
//! numbered text it must leave alone.
use super::*;

const NEWLINE: u32 = 709;
const SPACE: u32 = 731;
const PERIOD: u32 = 537;
const TR: [u32; 2] = [131, 132];
const TD: [u32; 2] = [269, 270];
/// What the deep-fried page `ad43a4d35e59199e` writes after each number in
/// its counting loop: ` 女 (四、刃 (四、刃)` and a blank line.
const ROW: [u32; 16] = [
    731, 4083, 731, 531, 3763, 911, 949, 736, 731, 531, 3763, 911, 949, 736, 532, 864,
];

/// Tokens kept when the stop fires, or None.
fn fires_after(tokens: &[u32]) -> Option<usize> {
    let mut stop = RepetitionStop::new();
    let mut generated = Vec::new();
    for &t in tokens {
        generated.push(t);
        if stop.push(&generated) {
            return Some(generated.len());
        }
    }
    None
}

fn with_loop(prefix: usize, cycle: &[u32], total: usize) -> Vec<u32> {
    let mut t: Vec<u32> = (1000..1000 + prefix as u32).collect();
    while t.len() < total {
        t.push(cycle[(t.len() - prefix) % cycle.len()]);
    }
    t
}

/// The digit tokens of `n`.
fn digits(n: u64) -> Vec<u32> {
    n.to_string()
        .bytes()
        .map(|b| DIGIT_ZERO + u32::from(b - b'0'))
        .collect()
}

/// `prefix` distinct tokens, then one row per number: its digits and `row`.
fn numbered(prefix: usize, numbers: impl IntoIterator<Item = u64>, row: &[u32]) -> Vec<u32> {
    let mut t: Vec<u32> = (1000..1000 + prefix as u32).collect();
    for n in numbers {
        t.extend(digits(n));
        t.extend_from_slice(row);
    }
    t
}

/// A pseudo-random word token per index, never equal for indices less than
/// 3,000 apart.
fn word(i: u64) -> u32 {
    5000 + (i * 1919 % 3000) as u32
}

#[test]
fn short_cycle_stops_once_the_window_is_full() {
    // Period 3 starting after 10 distinct tokens: the periodic window is
    // the loop itself, so the stop fires after 256 loop tokens.
    let tokens = with_loop(10, &[1, 2, 3], 1000);
    assert_eq!(fires_after(&tokens), Some(10 + MIN_WINDOW));
    assert_eq!(fires_after(&tokens[..10 + MIN_WINDOW - 1]), None);
}

#[test]
fn long_cycle_needs_four_repeats() {
    let cycle: Vec<u32> = (0..100).collect();
    let tokens = with_loop(5, &cycle, 1000);
    assert_eq!(fires_after(&tokens), Some(5 + MIN_REPEATS * 100));
}

#[test]
fn ignores_cycles_longer_than_the_limit_and_plain_text() {
    let cycle: Vec<u32> = (0..MAX_PERIOD as u32 + 1).collect();
    assert_eq!(fires_after(&with_loop(0, &cycle, 4000)), None);
    let text: Vec<u32> = (0..4000).map(|i| (i * 7919 % 4093) as u32).collect();
    assert_eq!(fires_after(&text), None);
}

#[test]
fn a_break_in_the_cycle_restarts_the_count() {
    let mut tokens = with_loop(0, &[7, 8], 200);
    tokens.push(9);
    tokens.extend(with_loop(0, &[7, 8], 255));
    assert_eq!(fires_after(&tokens), None);
    tokens.push(tokens[tokens.len() - 2]);
    assert_eq!(fires_after(&tokens), Some(tokens.len()));
}

#[test]
fn exact_loops_with_numbers_stop_where_they_did() {
    // `12 x` repeated: the numbers never move, so only the exact rule
    // applies, at its usual window.
    let cycle = [DIGIT_ZERO + 1, DIGIT_ZERO + 2, SPACE, 4000, NEWLINE];
    assert_eq!(fires_after(&with_loop(10, &cycle, 1000)), Some(10 + MIN_WINDOW));
}

/// The rows of `numbers`, as `numbered` writes them with `row`, in tokens.
fn rows_len(numbers: &[u64], row: &[u32]) -> usize {
    numbers.iter().map(|&n| digits(n).len() + row.len()).sum()
}

#[test]
fn counting_loop_stops_after_the_minimum_rows() {
    // Rows 44, 45, ...: each a number and 16 tokens, a cycle of 17 symbols
    // (18 tokens, 19 from row 100). The window starts at row 44 and holds
    // MIN_COUNT_REPEATS whole rows at the blank line that closes row 107.
    let tokens = numbered(10, 44..400, &ROW);
    let rows: Vec<u64> = (44..44 + MIN_COUNT_REPEATS as u64).collect();
    assert_eq!(rows.last(), Some(&107));
    let expected = 10 + rows_len(&rows, &ROW);
    assert!(expected - 10 >= MIN_WINDOW);
    assert_eq!(fires_after(&tokens), Some(expected));
    // The decision depends only on the tokens so far: one token less does
    // not stop, and what follows the stop cannot move it.
    assert_eq!(fires_after(&tokens[..expected - 1]), None);
    let mut other = tokens[..expected].to_vec();
    other.extend(with_loop(0, &[1, 2], 300));
    assert_eq!(fires_after(&other), Some(expected));
}

#[test]
fn counting_loops_may_step_by_any_constant_and_change_width() {
    // Counting down by 5 from 1,010 crosses from four digits to three.
    let numbers: Vec<u64> = (0..200).map(|i| 1010 - 5 * i).collect();
    let tokens = numbered(10, numbers.iter().copied(), &ROW);
    let kept = rows_len(&numbers[..MIN_COUNT_REPEATS], &ROW);
    assert_eq!(fires_after(&tokens), Some(10 + kept));
    // Several numbers per row, each with its own step: `n. 7 × n = 7n`.
    let times: Vec<u32> = (1..400)
        .flat_map(|n: u64| {
            let mut row = digits(n);
            row.extend([PERIOD, SPACE]);
            row.extend(digits(7));
            row.extend([24276, SPACE]);
            row.extend(digits(n));
            row.extend([SPACE, 552, SPACE]);
            row.extend(digits(7 * n));
            row.push(NEWLINE);
            row
        })
        .collect();
    assert!(fires_after(&times).is_some());
}

#[test]
fn a_number_off_the_step_restarts_the_count() {
    // Rows 44 to 49, then 49 again: the repeated number breaks the step, so
    // the window starts again with the text after the first 49 (row 5).
    // It holds MIN_COUNT_REPEATS whole cycles once the number of row 69
    // (112) is complete, at the token after its three digits.
    let numbers: Vec<u64> = (44..50).chain(49..400).collect();
    let tokens = numbered(10, numbers.iter().copied(), &ROW);
    let last = 5 + MIN_COUNT_REPEATS;
    assert_eq!((last, numbers[last]), (69, 112));
    let expected = 10 + rows_len(&numbers[..last], &ROW) + digits(numbers[last]).len() + 1;
    assert_eq!(fires_after(&tokens), Some(expected));
}

#[test]
fn numbered_lists_with_varying_items_never_stop() {
    // `n. word` for 1,000 lines: the numbers count, but the items differ.
    let list: Vec<u32> = (1..1000)
        .flat_map(|n| {
            let mut line = digits(n);
            line.extend([PERIOD, word(n), NEWLINE]);
            line
        })
        .collect();
    assert_eq!(fires_after(&list), None);
    // Line-numbered code: a number, then a line of one to four tokens.
    let code: Vec<u32> = (1..1000)
        .flat_map(|n| {
            let mut line = digits(n);
            line.push(SPACE);
            line.extend((0..1 + n % 4).map(|k| word(n * 4 + k)));
            line.push(NEWLINE);
            line
        })
        .collect();
    assert_eq!(fires_after(&code), None);
}

#[test]
fn tables_with_row_numbers_and_varying_cells_never_stop() {
    // `<tr><td>n</td><td>cell</td></tr>` one tag per line, as the model
    // writes tables, with a numeric cell that does not move by a fixed step
    // and with a text cell.
    let table = |cell: &dyn Fn(u64) -> Vec<u32>| -> Vec<u32> {
        (1..1000)
            .flat_map(|n| {
                let mut row = vec![TR[0], NEWLINE, TD[0]];
                row.extend(digits(n));
                row.extend([TD[1], NEWLINE, TD[0]]);
                row.extend(cell(n));
                row.extend([TD[1], NEWLINE, TR[1], NEWLINE]);
                row
            })
            .collect()
    };
    assert_eq!(fires_after(&table(&|n| digits(n * n * 7 % 1000 + 3))), None);
    assert_eq!(fires_after(&table(&|n| vec![word(n)])), None);
    // Rows identical apart from the row number are a counting loop: a blank
    // numbered form ends with its row MIN_COUNT_REPEATS, which takes the
    // window past MIN_WINDOW tokens.
    let blank = table(&|_| Vec::new());
    let kept: usize = (1..=MIN_COUNT_REPEATS as u64).map(|n| 10 + digits(n).len()).sum();
    assert!(kept >= MIN_WINDOW);
    assert_eq!(fires_after(&blank), Some(kept));
}

/// `n. A B C D` for items 1 to `items`, one line each, as on an answer
/// sheet.
fn answer_sheet(items: u64) -> Vec<u32> {
    let (a, b, c, d) = (4000, 4001, 4002, 4003);
    (1..=items)
        .flat_map(|n| {
            let mut line = digits(n);
            line.extend([PERIOD, SPACE, a, SPACE, b, SPACE, c, SPACE, d, NEWLINE]);
            line
        })
        .collect()
}

#[test]
fn templated_rows_with_fixed_steps_are_not_loops() {
    // A Celsius to Fahrenheit table, 0 to 200 °C in steps of 5 (41 rows,
    // one tag per line, as the model writes tables): both cells count, by 5
    // and by 9. At 32 cycles the stop cut it after row 32 (token 482).
    let conversion: Vec<u32> = (0..=40)
        .flat_map(|k: u64| {
            let mut row = vec![TR[0], NEWLINE, TD[0]];
            row.extend(digits(5 * k));
            row.extend([TD[1], NEWLINE, TD[0]]);
            row.extend(digits(32 + 9 * k));
            row.extend([TD[1], NEWLINE, TR[1], NEWLINE]);
            row
        })
        .collect();
    assert!(conversion.len() >= MIN_WINDOW);
    assert_eq!(fires_after(&conversion), None);
    // A 40-item answer sheet; 32 cycles cut it after item 32 (token 375).
    assert_eq!(fires_after(&answer_sheet(40)), None);
    // A 52-week planner, `Week n ____`; 32 cycles cut it at token 256.
    let (week, blank) = (4010, 4011);
    let planner: Vec<u32> = (1..=52)
        .flat_map(|n| {
            let mut line = vec![week, SPACE];
            line.extend(digits(n));
            line.extend([SPACE, blank, NEWLINE]);
            line
        })
        .collect();
    assert!(planner.len() >= MIN_WINDOW);
    assert_eq!(fires_after(&planner), None);
}

#[test]
fn a_counting_window_stops_at_64_cycles_not_63() {
    // The recorded counting loops reach 65 cycles and more. Both windows
    // span well over MIN_WINDOW tokens, so only the cycle count decides.
    assert_eq!(MIN_COUNT_REPEATS, 64);
    let (short, full) = (answer_sheet(63), answer_sheet(64));
    assert!(short.len() >= 2 * MIN_WINDOW);
    assert_eq!(fires_after(&short), None);
    assert_eq!(fires_after(&full), Some(full.len()));
}

#[test]
fn short_counting_runs_do_not_stop() {
    // A times table of 12 rows, and one row less than MIN_COUNT_REPEATS of
    // the counting loop.
    let times: Vec<u32> = (1..=12)
        .flat_map(|n: u64| {
            let mut row = digits(7);
            row.extend([24276, SPACE]);
            row.extend(digits(n));
            row.extend([SPACE, 552, SPACE]);
            row.extend(digits(7 * n));
            row.push(NEWLINE);
            row
        })
        .collect();
    assert_eq!(fires_after(&times), None);
    assert_eq!(
        fires_after(&numbered(10, 44..44 + MIN_COUNT_REPEATS as u64 - 1, &ROW)),
        None
    );
    // The longest counting run in legitimate output on record: a 20-row
    // scoring table (bonus points 20 down to 1, and twice that in each of
    // six grade columns), in OmniDocBench and in the v3 calibration pages.
    let scores: Vec<u32> = (1..=20)
        .rev()
        .flat_map(|n: u64| {
            let mut row = vec![TR[0], TD[0]];
            row.extend(digits(n));
            row.push(TD[1]);
            for _ in 0..6 {
                row.push(TD[0]);
                row.extend(digits(2 * n));
                row.push(TD[1]);
            }
            row.extend([TR[1], NEWLINE]);
            row
        })
        .collect();
    assert!(scores.len() >= MIN_WINDOW);
    assert_eq!(fires_after(&scores), None);
}
