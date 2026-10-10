"""Bring frozen canister sources up to the current stable receipt contract."""
import shutil

PAID_SOURCES = ('paid_inference.rs', 'paid_types.rs', 'receipt_archive.rs')
OLD_FOOTER = 'ic_cdk::api::stable_write(total * 65536 - 8, &end.to_le_bytes());'
CURRENT_FOOTER = 'ic_cdk::api::stable_write(ic_cdk::api::stable_size() * 65536 - 8, &end.to_le_bytes());'
METADATA_END = 'let end = (m.bytes + 65535) / 65536 * 65536;'
METADATA_BOUND = 'assert!(metadata.len() < 2_000_000, "upgrade metadata size");'


def copy_paid_sources(directory, canonical):
    # Check the complete set before copying, so a missing dependency fails early.
    for name in PAID_SOURCES:
        if not (canonical / name).is_file():
            raise ValueError(f'missing current paid source: {name}')
    for name in PAID_SOURCES:
        shutil.copyfile(canonical / name, directory / name)


def align_upgrade_metadata(directory):
    path = directory / 'lib.rs'
    text = path.read_text()
    old, current = text.count(OLD_FOOTER), text.count(CURRENT_FOOTER)
    if (old, current) not in ((1, 0), (0, 1)):
        raise ValueError('unexpected upgrade metadata footer')
    if text.count(METADATA_END) != 1 or text.count(METADATA_BOUND) > 1:
        raise ValueError('unexpected upgrade metadata bounds')
    if METADATA_BOUND in text and METADATA_BOUND + '\n            ' + METADATA_END not in text:
        raise ValueError('unexpected upgrade metadata bound position')
    text = text.replace(OLD_FOOTER, CURRENT_FOOTER)
    if METADATA_BOUND not in text:
        text = text.replace(METADATA_END, METADATA_BOUND + '\n            ' + METADATA_END)
    path.write_text(text)
