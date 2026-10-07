const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const root = path.resolve(__dirname, '..');
const directory = path.join(root, 'artifacts/sha256-batch-real-fixtures-v1');
const reportPath = path.join(directory, 'report.json');
const report = JSON.parse(fs.readFileSync(reportPath));
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const assert = (condition, context) => { if (!condition) throw Error(context); };
assert(report.complete && report.groups.length === 9, 'fixture report incomplete');
for (const [relative, expected] of Object.entries(report.source_hashes)) {
  assert(digest(fs.readFileSync(path.join(root, relative))) === expected, relative);
}
let count = 0;
for (const group of report.groups) {
  assert(group.messages.length === 4, 'four messages required');
  for (const item of group.messages) {
    const bytes = fs.readFileSync(path.join(root, item.path));
    assert(bytes.length === item.bytes && item.bytes === group.bytes_per_message, item.path);
    assert(digest(bytes) === item.sha256, item.path);
    count++;
  }
}
assert(count === 36, 'message count');
const result = {
  complete: true, messages: count, all_raw_hashes_exact: true,
  fixture_report_sha256: digest(fs.readFileSync(reportPath)),
  verifier_sha256: digest(fs.readFileSync(__filename)),
  scope: 'Independent Node crypto hashes of all36 raw fixture byte streams and all source identities. No SIMD, IC or performance claim.'
};
fs.writeFileSync(path.join(directory, 'independent-bytes-audit.json'), JSON.stringify(result, null, 2) + '\n');
console.log('PASS 36 independent raw message SHA256 checks');
