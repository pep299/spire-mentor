//! Wiki-image path resolution for cards and monsters.
//!
//! The sister `sts2-wiki` repo ships a complete sprite library under
//! `site/public/images/`:
//!
//! - `card_atlas/{character}/{snake_card_name}.png` (850 files)
//! - `monsters/{snake_monster_name}.png`            (100 files)
//!
//! Card character folder comes from `cards::character_of(card_id)`
//! (already a snake-case string like `"ironclad"`). The image stem
//! is `CardId::to_python_name()` lowercased — Python parity names
//! are `UPPER_SNAKE_CASE` and the wiki uses `lower_snake_case`.
//!
//! Wiki root defaults to the monorepo sibling `sts2-wiki`, located by walking
//! up from the working directory (so the tool works from the repo root, from
//! `sts2-rs/`, or from a worktree) and is overridable via the `STS2_WIKI` env
//! var. Both helpers return
//! `None` when the file does not exist on disk — the HTML renderer
//! falls back to a text badge in that case so screenshots stay
//! debuggable when an asset is missing.

use std::path::{Path, PathBuf};

use sts2_core::cards::{character_of, CardId};
use sts2_core::monsters::MonsterKind;

const WIKI_DIR_NAME: &str = "sts2-wiki";

/// Root of the sibling `sts2-wiki` checkout.
///
/// `STS2_WIKI` wins when set. Otherwise walk up from the working directory
/// looking for a `sts2-wiki` sibling — a bare relative `../sts2-wiki` only
/// resolves from exactly one cwd, and silently degrades every sprite to a text
/// badge from any other. Falls back to the sibling of this crate's source tree
/// (baked at compile time) when the cwd is outside the monorepo.
#[must_use]
pub fn wiki_root() -> PathBuf {
    if let Some(explicit) = std::env::var_os("STS2_WIKI") {
        return PathBuf::from(explicit);
    }
    if let Ok(cwd) = std::env::current_dir() {
        for dir in cwd.ancestors() {
            let candidate = dir.join(WIKI_DIR_NAME);
            if candidate.is_dir() {
                return candidate;
            }
        }
    }
    // `<repo>/sts2-rs/crates/sts2-cli` -> `<repo>/sts2-wiki`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .map_or_else(
            || PathBuf::from(WIKI_DIR_NAME),
            |repo| repo.join(WIKI_DIR_NAME),
        )
}

#[must_use]
pub fn card_image_path(card_id: CardId) -> Option<PathBuf> {
    let stem_raw = card_id.to_python_name().to_ascii_lowercase();
    // The Python ID retains a trailing `_card` on a handful of
    // cards (e.g. `FEEL_NO_PAIN_CARD`, `CRUELTY_CARD`) where the
    // sister enum needed to disambiguate from a power id. The wiki
    // strips it on disk.
    let stem_trim = stem_raw
        .strip_suffix("_card")
        .map_or_else(|| stem_raw.clone(), str::to_string);
    let primary = character_of(card_id);
    // `character_of` returns the gameplay-pool ("colorless", "curse",
    // "status", "ironclad", …) which is *usually* the wiki folder
    // — but curses, statuses, and event-source cards live under
    // dedicated folders even when their gameplay color is colorless.
    // Try the primary folder first, then fall back through the
    // dedicated content folders.
    let atlas = wiki_root().join("site/public/images/card_atlas");
    let folders = ["curse", "status", "event", "token", "quest"];
    for stem in [stem_trim.as_str(), stem_raw.as_str()] {
        if let Some(p) = resolve(atlas.join(primary).join(format!("{stem}.png"))) {
            return Some(p);
        }
        for folder in folders {
            if let Some(p) = resolve(atlas.join(folder).join(format!("{stem}.png"))) {
                return Some(p);
            }
        }
    }
    None
}

#[must_use]
pub fn monster_image_path(kind: MonsterKind) -> Option<PathBuf> {
    // `MonsterKind::python_name` returns PascalCase (e.g. `BowlbugEgg`)
    // — the wiki uses lower_snake_case. Card IDs already come out
    // UPPER_SNAKE so they snake-case via `to_ascii_lowercase` alone;
    // monsters need an explicit camel→snake conversion.
    let stem = pascal_to_snake(kind.python_name());
    let dir = wiki_root().join("site/public/images/monsters");
    if let Some(p) = resolve(dir.join(format!("{stem}.png"))) {
        return Some(p);
    }
    // Wiki occasionally collapses size / form suffixes into a shared
    // sprite — e.g. `BOWLBUG_EGG` and `BOWLBUG_ROCK` both map to
    // `bowlbug.png`, and `LEAF_SLIME_S` / `LEAF_SLIME_M` to
    // `leaf_slime.png`. Try progressively trimming `_xxx` suffixes
    // until we either hit a sprite or run out of underscores.
    let mut trimmed: &str = &stem;
    while let Some((head, _)) = trimmed.rsplit_once('_') {
        trimmed = head;
        if let Some(p) = resolve(dir.join(format!("{trimmed}.png"))) {
            return Some(p);
        }
    }
    None
}

fn resolve(path: PathBuf) -> Option<PathBuf> {
    if path.is_file() {
        Some(path)
    } else {
        None
    }
}

/// `file://`-prefixed absolute path string suitable for `<img src>`.
///
/// The path is absolutized first. A relative one would render as
/// `file://../sts2-wiki/…`, which a browser parses as scheme `file` with
/// authority `..` — a broken image, and worse than the missing-asset text
/// badge because nothing falls back.
///
/// Windows needs three more corrections, all of which produced an unloadable
/// `<img src>` before: `canonicalize` returns an extended-length `\\?\C:\…`
/// path, every separator is a backslash, and an absolute path starts with a
/// drive letter rather than `/` — so the naive format yielded
/// `file://\\?\C:\…`, whose authority is `\\?\C:\…` and whose path is empty.
/// The canonical form is `file:///C:/…`.
///
/// The path is then percent-encoded (GH #437): a wiki root under a directory
/// with a space (`C:\Program Files\…`, a user dir like `John Doe`), a `#` or
/// a `?` otherwise yields a URL the browser cuts short or reads as a fragment
/// or query, and the image silently fails like the cases above.
#[must_use]
pub fn file_url(path: &Path) -> String {
    let abs = path.canonicalize().unwrap_or_else(|_| {
        std::env::current_dir().map_or_else(|_| path.to_path_buf(), |cwd| cwd.join(path))
    });
    let slashed = percent_encode_path(&url_path_bytes(&abs));
    if slashed.starts_with('/') {
        format!("file://{slashed}")
    } else {
        // Drive-rooted (`C:/…`): the URL's empty authority needs its own slash.
        format!("file:///{slashed}")
    }
}

/// The bytes of an absolute path as a `/`-separated URL path, before
/// percent-encoding.
///
/// Unix: the path's own bytes, unchanged. A path need not be UTF-8 there, and
/// `display()` would turn each stray byte into U+FFFD (`%EF%BF%BD`), a URL to
/// a file that does not exist; a backslash is an ordinary filename byte and
/// stays one (`%5C`).
#[cfg(unix)]
fn url_path_bytes(abs: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt as _;
    abs.as_os_str().as_bytes().to_vec()
}

/// Windows: strip the extended-length prefix and turn separators into `/`.
/// `\\?\UNC\server\share` is the verbatim spelling of `\\server\share`; it
/// maps back onto the `//server/share` URL path, so it keeps two slashes where
/// the plain `\\?\C:\…` form keeps none.
#[cfg(not(unix))]
fn url_path_bytes(abs: &Path) -> Vec<u8> {
    let disp = abs.display().to_string();
    let stripped = disp.strip_prefix(r"\\?\UNC\").map_or_else(
        || disp.strip_prefix(r"\\?\").unwrap_or(&disp).to_string(),
        |unc| format!("//{unc}"),
    );
    stripped.replace('\\', "/").into_bytes()
}

/// Percent-encode a `/`-separated path for a URL: every byte except RFC 3986
/// unreserved characters, `/` and `:` (the drive letter's colon) becomes
/// `%XX`. Over-encoding a sub-delimiter is harmless; a `file://` path is
/// percent-decoded back to the filesystem path either way.
fn percent_encode_path(path: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(path.len());
    for &b in path {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~' | b'/' | b':') {
            out.push(char::from(b));
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

fn pascal_to_snake(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for (i, ch) in s.chars().enumerate() {
        if ch.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        for c in ch.to_lowercase() {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CTR: AtomicUsize = AtomicUsize::new(0);

    /// A fresh temp wiki root (unique per call) with the
    /// `site/public/images/{card_atlas,monsters}` skeleton, pointed at by
    /// `STS2_WIKI`. nextest runs one process per test, so the env var
    /// cannot leak across tests.
    fn tmp_wiki_root() -> PathBuf {
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("sts2assets_{}_{n}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("site/public/images/card_atlas")).unwrap();
        fs::create_dir_all(root.join("site/public/images/monsters")).unwrap();
        std::env::set_var("STS2_WIKI", &root);
        root
    }

    #[test]
    fn pascal_to_snake_inserts_underscores_before_inner_capitals() {
        assert_eq!(pascal_to_snake("BowlbugEgg"), "bowlbug_egg");
        assert_eq!(pascal_to_snake("Cultist"), "cultist");
        // No leading underscore for the first capital.
        assert_eq!(pascal_to_snake("A"), "a");
        assert_eq!(pascal_to_snake(""), "");
    }

    /// The invariants an `<img src>` actually needs, on every platform: empty
    /// authority (three slashes), no backslash separators, and no leftover
    /// Windows extended-length `\\?\` prefix.
    fn assert_loadable_file_url(url: &str) {
        assert!(url.starts_with("file:///"), "not a rooted file URL: {url}");
        assert!(!url.contains('\\'), "backslash separator in URL: {url}");
        assert!(!url.contains("//?/"), "verbatim prefix leaked: {url}");
    }

    #[test]
    fn file_url_prefixes_path_with_scheme() {
        let url = file_url(Path::new("/x/y/z.png"));
        assert_loadable_file_url(&url);
        assert!(url.ends_with("/x/y/z.png"), "path not preserved: {url}");
        #[cfg(unix)]
        assert_eq!(url, "file:///x/y/z.png");
        // On Windows a rooted-but-driveless path is resolved against the cwd's
        // drive, so the exact string is `file:///C:/x/y/z.png` — the leading
        // slash before the drive letter is what makes the authority empty.
        #[cfg(windows)]
        {
            let rest = &url["file:///".len()..];
            assert!(
                rest.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
                    && rest[1..].starts_with(":/"),
                "expected a drive-rooted URL: {url}"
            );
        }
    }

    #[test]
    fn file_url_percent_encodes_spaces_and_url_delimiters() {
        // GH #437: a raw space, `#` or `?` makes the browser cut the path
        // short (or read the rest as a fragment / query), so the art fails.
        let url = file_url(Path::new("/Program Files/a#b?c/d%e/caf\u{e9}.png"));
        assert_loadable_file_url(&url);
        assert!(
            url.ends_with("/Program%20Files/a%23b%3Fc/d%25e/caf%C3%A9.png"),
            "not percent-encoded: {url}"
        );
        assert!(!url.contains(' '), "raw space in URL: {url}");
    }

    #[test]
    fn percent_encode_path_keeps_drive_colon_and_separators() {
        assert_eq!(percent_encode_path(b"C:/x y/z.png"), "C:/x%20y/z.png");
        assert_eq!(percent_encode_path(b"/a-b_c.d~e/F9"), "/a-b_c.d~e/F9");
    }

    /// A Unix path need not be UTF-8: its raw bytes are encoded, not the
    /// U+FFFD that `display()` substitutes, and a backslash in a file name is
    /// a byte like any other rather than a separator.
    #[cfg(unix)]
    #[test]
    fn file_url_encodes_raw_unix_path_bytes() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt as _;
        let url = file_url(Path::new(OsStr::from_bytes(b"/home/jos\xe9/a\\b.png")));
        assert_loadable_file_url(&url);
        assert_eq!(url, "file:///home/jos%E9/a%5Cb.png");
    }

    #[test]
    fn file_url_absolutizes_relative_paths() {
        // A relative path must never reach the browser as `file://../…`
        // (authority `..`); it is resolved against the cwd first.
        let url = file_url(Path::new("../sts2-wiki/x.png"));
        assert_loadable_file_url(&url);
        assert!(!url.starts_with("file://.."), "authority-relative: {url}");
    }

    #[test]
    fn file_url_of_an_existing_file_is_loadable() {
        // The path that EXISTS is the one `canonicalize` succeeds on — the
        // branch that returns `\\?\C:\…` on Windows, which the naive
        // `format!("file://{}")` turned into a URL whose authority was the
        // whole path. Every real asset lookup takes this branch.
        let root = tmp_wiki_root();
        let png = root.join("site/public/images/monsters/cultist.png");
        fs::write(&png, b"x").unwrap();
        let url = file_url(&png);
        assert_loadable_file_url(&url);
        assert!(url.ends_with("/cultist.png"), "path not preserved: {url}");
    }

    #[test]
    fn wiki_root_honors_env_override() {
        let root = tmp_wiki_root();
        assert_eq!(wiki_root(), root);
    }

    #[test]
    fn wiki_root_default_is_absolute() {
        // Without `STS2_WIKI` the default must be usable from any cwd — an
        // absolute path, never a bare `../sts2-wiki`.
        std::env::remove_var("STS2_WIKI");
        assert!(wiki_root().is_absolute(), "{:?} is relative", wiki_root());
    }

    #[test]
    fn card_image_path_resolves_in_primary_folder_and_misses_when_absent() {
        let root = tmp_wiki_root();
        let card = CardId::TwinStrike;
        // None until the sprite exists on disk.
        assert!(card_image_path(card).is_none());

        // Reconstruct the primary-folder lookup the resolver tries first.
        let stem_raw = card.to_python_name().to_ascii_lowercase();
        let stem = stem_raw
            .strip_suffix("_card")
            .map_or(stem_raw.clone(), str::to_string);
        let dir = root
            .join("site/public/images/card_atlas")
            .join(character_of(card));
        fs::create_dir_all(&dir).unwrap();
        let expected = dir.join(format!("{stem}.png"));
        fs::write(&expected, b"png").unwrap();

        assert_eq!(card_image_path(card), Some(expected));
    }

    #[test]
    fn monster_image_path_resolves_exact_sprite() {
        let root = tmp_wiki_root();
        let kind = MonsterKind::Nibbit;
        assert!(monster_image_path(kind).is_none());

        let stem = pascal_to_snake(kind.python_name());
        let p = root
            .join("site/public/images/monsters")
            .join(format!("{stem}.png"));
        fs::write(&p, b"png").unwrap();
        assert_eq!(monster_image_path(kind), Some(p));
    }

    #[test]
    fn monster_image_path_falls_back_through_suffix_trim() {
        let root = tmp_wiki_root();
        // `BowlbugEgg` -> `bowlbug_egg`; only the collapsed `bowlbug.png`
        // sprite exists, so the resolver must trim `_egg` and hit it.
        let shared = root.join("site/public/images/monsters").join("bowlbug.png");
        fs::write(&shared, b"png").unwrap();
        assert_eq!(monster_image_path(MonsterKind::BowlbugEgg), Some(shared));
    }
}
