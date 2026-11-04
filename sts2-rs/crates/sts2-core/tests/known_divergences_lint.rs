//! Schema lint for `tests/parity/known_divergences.md`.
//!
//! Bead sts2-rs-pmf (P6-1b) tightens the divergence-manifest schema (D13 of
//! the synthesized P6 plan). Every entry — defined by a `## YYYY-MM-DD `
//! heading — must declare:
//!
//! * **Status:** one of `investigated | accepted-permanently |
//!   deferred-to-bead-<id>`.
//! * **Bead:** required iff Status starts with `deferred-`.
//! * **Expires:** required iff Status is `investigated` or starts with
//!   `deferred-`. Must parse as `YYYY-MM-DD` and not be in the past.
//!
//! Hand-rolled regex parser — no new build deps (per ROBUSTNESS §9 #3).
//!
//! The lint accepts the today-file as the empty-entries baseline. As soon
//! as a real entry lands (post-shrinker, post-triage), the schema is
//! enforced.

use std::fs;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates parent")
        .parent()
        .expect("repo root")
        .to_path_buf()
}

/// Returns `(yyyy, mm, dd)` if `s` parses as a calendar date.
#[allow(clippy::many_single_char_names)]
fn parse_date(s: &str) -> Option<(u32, u32, u32)> {
    if s.len() != 10 {
        return None;
    }
    let bytes = s.as_bytes();
    if bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    let year: u32 = s[0..4].parse().ok()?;
    let month: u32 = s[5..7].parse().ok()?;
    let day: u32 = s[8..10].parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some((year, month, day))
}

/// 2026-05-12 — the bead's authored date. Used as a coarse "today" lower
/// bound; we don't link against `chrono` for one comparison. Bumps when
/// the lint test is updated for a new year.
const TODAY: (u32, u32, u32) = (2026, 5, 12);

fn date_in_past(d: (u32, u32, u32)) -> bool {
    d < TODAY
}

struct Entry<'a> {
    slug: &'a str,
    body: &'a str,
}

fn parse_entries(src: &str) -> Vec<Entry<'_>> {
    // Skip the schema example block: anything between ```...``` fences is
    // ignored.
    let mut out: Vec<Entry<'_>> = Vec::new();
    let mut in_code = false;
    let mut current_slug: Option<&str> = None;
    let mut current_start: usize = 0;
    let mut byte_pos: usize = 0;

    for line in src.split_inclusive('\n') {
        let line_start = byte_pos;
        byte_pos += line.len();
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
            continue;
        }
        if in_code {
            continue;
        }
        let trimmed = line.trim_end();
        // Heading marker: `## <date> <slug>` (the schema's literal `<YYYY-...>`
        // placeholder in the schema block is excluded because we're past
        // the code fence).
        if let Some(rest) = trimmed.strip_prefix("## ") {
            if rest.starts_with('<') {
                continue; // schema placeholder
            }
            // Flush previous entry.
            if let Some(slug) = current_slug.take() {
                out.push(Entry {
                    slug,
                    body: &src[current_start..line_start],
                });
            }
            current_slug = Some(rest);
            current_start = byte_pos;
        }
    }
    if let Some(slug) = current_slug {
        out.push(Entry {
            slug,
            body: &src[current_start..],
        });
    }
    out
}

fn field(body: &str, name: &str) -> Option<String> {
    let needle = format!("**{name}:**");
    body.lines().find_map(|l| {
        l.find(&needle)
            .map(|i| l[i + needle.len()..].trim().to_string())
    })
}

#[test]
fn known_divergences_schema_is_valid() {
    let path = repo_root()
        .join("tests")
        .join("parity")
        .join("known_divergences.md");
    let src = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

    let entries = parse_entries(&src);
    // Filter out heading-shaped lines that aren't real entries (the
    // `## Entries` / `## Triage rule` / `## Schema` headings).
    let real_entries: Vec<&Entry<'_>> = entries
        .iter()
        .filter(|e| {
            // Real entries lead with a YYYY-MM-DD prefix.
            e.slug.len() >= 10 && parse_date(&e.slug[..10]).is_some()
        })
        .collect();

    let mut errors: Vec<String> = Vec::new();
    for e in &real_entries {
        let status = field(e.body, "Status").unwrap_or_default();
        if !(status == "investigated"
            || status == "accepted-permanently"
            || status.starts_with("deferred-to-bead-"))
        {
            errors.push(format!(
                "{}: Status must be 'investigated' | 'accepted-permanently' | 'deferred-to-bead-<id>' (got {status:?})",
                e.slug
            ));
            continue;
        }
        if status.starts_with("deferred-") && field(e.body, "Bead").is_none() {
            errors.push(format!("{}: deferred status requires Bead field", e.slug));
        }
        if status == "investigated" || status.starts_with("deferred-") {
            let exp = field(e.body, "Expires").unwrap_or_default();
            match parse_date(&exp) {
                None => errors.push(format!(
                    "{}: Expires must be YYYY-MM-DD (got {exp:?})",
                    e.slug
                )),
                Some(d) if date_in_past(d) => errors.push(format!(
                    "{}: Expires {exp} is in the past — re-triage",
                    e.slug
                )),
                _ => {}
            }
        }
    }

    assert!(
        errors.is_empty(),
        "known_divergences.md schema violations:\n{}",
        errors.join("\n")
    );
}
