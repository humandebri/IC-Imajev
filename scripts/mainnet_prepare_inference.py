#!/usr/bin/env python3
"""Offline-first plan / explicit mainnet owner cache bootstrap and UI smoke tests.

No funding, upload, identity export, PEM, relay or paid configuration mutations.
Run without --execute to validate all source assets and print a plan (no network).
Runtime receipts are resumable; unknown non-idempotent outcomes MUST stop. Do not
upgrade/reinstall or run another owner inference concurrently with this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TARGET = "xis3j-paaaa-aaaai-axumq-cai"
IDENTITY = "llm-wiki-mainnet"
LIVE_GZIP_SHA = "5e2a0b101d4dde0ff503a8c7d399efc17e7db7618dc2c3aa6eddee49a771b3a9"
RAW_SHA = "c80e77130301636411b6df230f7ce04892898c5ba694e242b4c390c6a8d746ff"
DID = ROOT / "canisters/inference/paid-inference.did"
DENSE_DID = ROOT / "artifacts/mainnet-deploy-20261007/prepare-fixed-prefix.did"
DENSE_SCHEMA = "service:{prepare_fixed_prefix_state:(vec nat8,vec nat8)->(variant {Ok:record {nat32;nat64;nat64};Err:text})}\n"
BANKS = (
    ("voting", 38, ROOT / "artifacts/voting-template-prefix-v1/prefix",
     ROOT / "artifacts/voting-template-prefix-v1/packets"),
    ("common", 27, ROOT / "artifacts/query-packing-v3/prefix-v2",
     ROOT / "artifacts/query-packing-v3/packets-v2"),
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path, value):
    """Atomic durable checkpoints; never write partial JSON as a final receipt."""
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def bank_values(bank, layer):
    _, tokens, source, _ = bank
    path = source / "queries/states" / f"layer-{layer:02d}.npz"
    with np.load(path, allow_pickle=False) as z:
        if layer % 4 == 3:
            keys, values = z["keys"], z["values"]
            require(keys.shape == values.shape == (tokens, 4, 256),
                    f"Unexpected KV shape: {path}")
            v = np.concatenate([keys.transpose(1, 0, 2).ravel(),
                                values.transpose(1, 0, 2).ravel()])
        else:
            require(z["conv"].shape == (3, 8192), f"Unexpected conv shape: {path}")
            require(z["delta_log"].size == tokens * 6176,
                    f"Unexpected delta log shape: {path}")
            v = z["conv"].ravel()
    v = np.asarray(v, dtype="<f4")
    require(np.isfinite(v).all() and np.all((v.view("<u4") & 65535) == 0),
            f"Prefix not exact finite BF16: {path}")
    return v


def offline_plan(args):
    """Bind original assets, not generated guesses or native expected answers."""
    manifest_path = ROOT / args.manifest
    manifest = load(manifest_path)
    model = sha(ROOT / "MODEL_LOCK.json")
    require(manifest["model"] == model, "Manifest/model lock mismatch")
    require(sha(ROOT / args.wasm) == RAW_SHA, "Frozen raw module hash mismatch")
    tensors = [x for x in manifest["tensors"]
               if x["dtype"] in ("int8", "f32") and "embed_tokens" not in x["name"]]
    require(len(tensors) == 721, "Expected exactly 721 dense INT8 + F32 tensors")
    require(len({x["name"] for x in tensors}) == len(tensors), "Duplicate weight names")
    for t in tensors:
        expected = t["rows"] * t["cols"] * (4 if t["dtype"] == "f32" else 1)
        if t["dtype"] == "int8":
            expected += 4 * t["rows"]
        require(t["bytes"] == expected and 0 < expected <= 128 * 1024**2,
                f"Invalid cache tensor: {t['name']}")
    weight_bytes = sum(x["bytes"] for x in tensors)
    require(weight_bytes <= 4_080_000_000, "Frozen dense cache budget exceeded")
    paired_bytes = sum(x["rows"] * x["cols"] for x in tensors
                       if x["dtype"] == "int8" and x["rows"] >= 32
                       and x["rows"] % 32 == 0 and x["cols"] % 256 == 0
                       and x["rows"] * x["cols"] <= 30_000_000)
    hashes = {}

    def bind(p):
        hashes[str(p.relative_to(ROOT))] = sha(p)

    for path in (manifest_path, ROOT / "MODEL_LOCK.json", DID, ROOT / args.wasm,
                 Path(__file__).resolve(), ROOT / args.prepared, ROOT / args.ui_results):
        bind(path)
    banks = []
    for bank in BANKS:
        name, tokens, source, packets = bank
        report = load(source / "report.json")
        cache = load(packets / "cache.json")
        identity = cache["identity"]
        require(report["model"] == identity["model"] == model and
                report["pack_hash"] == identity["pack_hash"] == manifest["pack_hash"] and
                report["tokens"] == identity["tokens"] == tokens,
                f"{name} source identity mismatch")
        require(identity["source_report_sha256"] == sha(source / "report.json"),
                f"{name} report hash mismatch")
        bind(source / "report.json")
        bind(packets / "cache.json")
        packet_bytes = 0
        for layer in range(32):
            path = source / "queries/states" / f"layer-{layer:02d}.npz"
            bind(path)
            bank_values(bank, layer)
            if layer % 4 != 3:
                require(sha(path) == identity["layers"][str(layer)],
                        f"{name} NPZ hash mismatch at layer {layer}")
                packet = packets / f"layer-{layer:02d}.npf1"
                raw = packet.read_bytes()
                entry = cache["packets"][str(layer)]
                require(sha(packet) == entry["sha256"] and len(raw) == entry["bytes"],
                        f"{name} NPF1 hash/length mismatch at layer {layer}")
                require(raw[:4] == b"NPF1" and int.from_bytes(raw[4:8], "little") == tokens,
                        f"{name} NPF1 token mismatch at layer {layer}")
                require(len(raw) + 3 * 8192 * 4 + 128 < 1_990_000,
                        f"{name} prefix request too large")
                bind(packet)
                packet_bytes += len(raw)
        banks.append(dict(name=name, tokens=tokens, layers=32, packet_bytes=packet_bytes,
                          value_bytes=24 * 3 * 8192 * 4 + 8 * tokens * 2048 * 4))

    # Same 24 frame logs + packets as prepare_fixed_prefix_states.py; bit-check NPZ.
    sys.path.insert(0, str(ROOT / "client"))
    from transport import decode  # offline binary frame decoder only; no Transport instance
    common_source, common_packets = BANKS[1][2:]
    report = load(common_source / "report.json")
    metrics = [q for q in report["queries"] if q.get("op") == "delta_full_log_integer"]
    require(len(metrics) == 24, "Expected 24 common prefix log frames")
    dense = []
    for q in metrics:
        layer = int(q["tensor"].split(".")[3])
        frame = common_source / "queries" / f"{q['index']:06d}.response.bin"
        header, values = decode(frame.read_bytes())
        require(header["dims"] == [27, 32, 0, 1], "Common log frame shape mismatch")
        log = np.asarray(values[27 * 2560 + 24576:], dtype="<f4")
        require(log.size == 27 * 6176, "Common prefix log length mismatch")
        with np.load(common_source / "queries/states" / f"layer-{layer:02d}.npz",
                     allow_pickle=False) as z:
            require(log.tobytes() == np.asarray(z["delta_log"], dtype="<f4").tobytes(),
                    f"Frame/NPZ log bit mismatch at layer {layer}")
        packet = common_packets / f"layer-{layer:02d}.npf1"
        require(log.nbytes + packet.stat().st_size + 128 < 1_990_000,
                "Dense prefix Candid request too large")
        bind(frame)
        dense.append(dict(layer=layer, frame=str(frame.relative_to(ROOT)),
                          log_sha256=hashlib.sha256(log.tobytes()).hexdigest()))
    require({x["layer"] for x in dense} == {i for i in range(32) if i % 4 != 3},
            "Common dense prefix layers mismatch")
    prepared = load(ROOT / args.prepared)
    ui = load(ROOT / args.ui_results)
    require(prepared["model_lock_sha256"] == model, "UI prepared model mismatch")
    records = prepared["records"]
    require(len(records) == len(ui["records"]) == 3, "Expected exactly three actual UI inputs")
    ui_by_id = {r["id"]: r for r in ui["records"]}
    examples = []
    prefix = None
    for record in records:
        ident = record["id"]
        require(re.fullmatch(r"[a-zA-Z0-9_-]+", ident), "Unsafe UI record id")
        ids = record["token_ids"]
        options = record["options"]
        require(all(type(x) is int and 0 <= x <= 0xFFFFFFFF for x in ids), "Bad UI token IDs")
        require(record["prefix_tokens"] == 27 and 1 <= len(ids) - 27 <= 89,
                "UI owner suffix bound/prefix mismatch")
        require(prefix is None or prefix == ids[:27], "UI inputs do not share the common prefix")
        prefix = ids[:27]
        reference = ui_by_id[ident]
        require(reference["tokens"] == dict(total=len(ids), prefix=27, suffix=len(ids) - 27)
                and reference["input"]["options"] == options, "UI input provenance mismatch")
        examples.append(dict(id=ident, token_ids=ids, prefix_tokens=27,
                             suffix_token_ids=ids[27:], options=options,
                             input_sha256=record["input_sha256"], input=reference["input"]))
    embed_request = common_source / "queries/000000.request.bin"
    embed_header, embed_values = decode(embed_request.read_bytes())
    require(embed_header["op"] == "embed" and embed_header["dims"] == [27, 2560]
            and embed_header["model"] == model and embed_header["pack_hash"] == manifest["pack_hash"],
            "Common prefix embedding source identity mismatch")
    require(embed_values.size == 27 and np.array_equal(embed_values, np.asarray(prefix, dtype=np.float32)),
            "Cached prefix token IDs differ from the current UI prompt")
    bind(embed_request)
    prepared_heap_bytes = weight_bytes + 24 * 2_097_152 + sum(
        b["packet_bytes"] + b["value_bytes"] for b in banks) + 131072 + 1048576
    require(prepared_heap_bytes < 4 * 1024**3 - args.heap_reserve_bytes,
            "Prepared caches leave insufficient configured 4GiB heap scratch margin")
    plan = dict(version=1, canister=TARGET, identity=IDENTITY, network="ic",
                live_gzip_module_sha256=LIVE_GZIP_SHA, raw_module_sha256=RAW_SHA,
                model=model, pack_hash=manifest["pack_hash"], pack_bytes=manifest["bytes"],
                weights=dict(count=len(tensors), bytes=weight_bytes, paired_weight_bytes=paired_bytes,
                             include_f32=True, require_prepared_rope=True,
                             require_prepared_activation=True, require_all_output_pairs=True),
                dense_prefix=dict(tokens=27, calls=24, bytes=24 * 2_097_152, layers=dense,
                                  candid_schema=DENSE_SCHEMA,
                                  candid_path=str(DENSE_DID.relative_to(ROOT))),
                prefix_banks=banks, examples=examples, source_hashes=hashes,
                bootstrap_update_calls=721 + 24 + 64, paid_enabled=False,
                prepared_heap_payload_bytes=prepared_heap_bytes,
                heap_payload_headroom_bytes=4 * 1024**3 - prepared_heap_bytes,
                heap_reserve_bytes=args.heap_reserve_bytes,
                memory_note="Payload estimate excludes allocator/graph/scratch overhead; check actual owner heap_pages after each stage.",
                floor_cycles=args.floor_cycles,
                warning="No owner progress query exists. Never blindly replay lost start/continue or prefix updates. "
                        "Voting is registered first, common last; owner start uses selected common bank. "
                        "Do not upgrade/reinstall or use concurrent writers while resuming.",
                scope="Preparation and actual UI owner updates only; no funding/upload/paid config or expected decisions")
    return plan, tensors


class Runner:
    def __init__(self, args, plan):
        from mainnet_cli import MainnetCLI
        self.cli = MainnetCLI(ROOT / args.directory,
                              codec=ROOT / args.codec, floor_cycles=args.floor_cycles)
        self.directory = ROOT / args.directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.plan = plan
        self.args = args
        self.path = self.directory / "preparation-state.json"
        digest = hashlib.sha256(canonical(plan).encode()).hexdigest()
        self.state = load(self.path) if self.path.exists() else dict(plan_sha256=digest, calls={})
        require(self.state["plan_sha256"] == digest, "Plan/assets/options changed since checkpoint")
        save(self.directory / "preparation-plan.json", plan)
        DENSE_DID.parent.mkdir(parents=True, exist_ok=True)
        if DENSE_DID.exists():
            require(DENSE_DID.read_text() == DENSE_SCHEMA, "Dense prefix DID changed")
        else:
            DENSE_DID.write_text(DENSE_SCHEMA)
        self.updated = 0

    def call(self, key, method, args, *, idempotent=False, growth_bytes=0, did=DID):
        if key in self.state["calls"]:
            return self.state["calls"][key]
        require(self.updated < self.args.max_new_updates, "Per-run update cap reached; safely resume later")
        # A fresh freeze+0.25T floor before EVERY new update is deliberately conservative.
        self.cli.guard(growth_bytes=growth_bytes)
        value = self.cli.call(method, args, key=key, idempotent=idempotent, did=did)
        result = self.cli.unwrap(value)
        # Persist even semantic failures in helper receipts; no subsequent requests after Err.
        self.state["calls"][key] = result
        save(self.path, self.state)
        self.updated += 1
        print(json.dumps(dict(key=key, completed_updates_this_run=self.updated)), flush=True)
        return result

    def query(self, method):
        return self.cli.call(method, [], query=True, did=DID)

    def verify_ready(self):
        status = self.cli.status()
        require(status["module_hash"].removeprefix("0x") == LIVE_GZIP_SHA,
                "Live installed GZIP module hash mismatch")
        memory_limit = status["settings"]["wasm_memory_limit"]
        require(int(str(memory_limit).replace("_", "")) == 4 * 1024**3,
                "Canister Wasm heap limit must remain exactly 4GiB")
        self.cli.guard()
        pack = self.query("pack_status")
        require(pack["ready"] and pack["model"] == self.plan["model"]
                and pack["pack_hash"] == self.plan["pack_hash"]
                and pack["bytes"] == pack["received"] == pack["hashed"] == self.plan["pack_bytes"],
                "Parent must complete sealed pack first")
        config = self.query("paid_config")
        require(config["enabled"] is False, "Paid inference must remain disabled")
        save(self.directory / "paid-config-observed.json", config)

    def weights(self, tensors):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        concurrency = self.args.warm_concurrency
        require(1 <= concurrency <= 8, "Warm concurrency must be between 1 and 8")
        sizes = {t["name"]: t["bytes"] for t in tensors}
        expected = set(sizes)

        def validate_cache(info, requested=None):
            names = info["names"]
            named = set(names)
            require(len(names) == len(named) and named <= expected and
                    info["bytes"] == sum(sizes[n] for n in named),
                    "Warm cache names/byte accounting mismatch")
            require(requested is None or requested in named,
                    f"Warm acknowledgement omitted requested tensor: {requested}")
            return named

        def confirm(required):
            # Query snapshots may lag acknowledged updates. Query ONLY, never rewarm
            # an acknowledged name merely because a subsequent query is older.
            for attempt in range(6):
                info = self.query("weight_cache_status")
                named = validate_cache(info)
                if required <= named:
                    return info
                if attempt < 5:
                    time.sleep(2**attempt)
            raise RuntimeError("Weight cache query failed bounded acknowledgement confirmation; "
                               "possible upgrade/cache reset: stop without replaying mutations")

        checkpoint_names = {t["name"] for i, t in enumerate(tensors)
                            if f"warm-{i:04d}" in self.state["calls"]}
        cache = confirm(checkpoint_names)
        cached = validate_cache(cache)
        pending = [(f"warm-{i:04d}", t) for i, t in enumerate(tensors)
                   if t["name"] not in cached]
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            while pending:
                capacity = self.args.max_new_updates - self.updated
                require(capacity > 0, "Per-run update cap reached; safely resume later")
                batch = pending[:min(concurrency, capacity)]
                pending = pending[len(batch):]
                # One conservative freeze+growth+floor check covers all requests
                # issued in this bounded batch; dense/prefix/inference stay serial.
                self.cli.guard(growth_bytes=sum(t["bytes"] for _, t in batch))
                futures = {executor.submit(self.cli.call, "warm_weights", [t["name"]],
                                           key=key, idempotent=True, did=DID): (key, t)
                           for key, t in batch}
                failures = []
                for future in as_completed(futures):
                    key, tensor = futures[future]
                    try:
                        info = self.cli.unwrap(future.result())
                        named = validate_cache(info, tensor["name"])
                        # Out-of-order replies describe different valid snapshots;
                        # never demand equality with a mutable aggregate set here.
                        cached.update(named)
                        self.state["calls"][key] = info
                        save(self.path, self.state)  # main thread only
                        self.updated += 1
                        print(json.dumps(dict(key=key, completed_updates_this_run=self.updated,
                                              warm_concurrency=concurrency)), flush=True)
                    except Exception as error:
                        failures.append((key, error))
                # Drain this already-issued <=8 batch and persist successful ACKs,
                # then stop. No subsequent batch launches after ANY error.
                if failures:
                    key, error = failures[0]
                    raise RuntimeError(f"Warm batch stopped at {key}: {error}; "
                                       f"{len(failures)} failed/unknown calls; receipts retained") from error
        final = confirm(expected)
        require(final["bytes"] == self.plan["weights"]["bytes"], "Dense/F32 weight cache incomplete")
        require(final["rope_bytes"] == 131072 and final["activation_bytes"] == 1048576 and
                final["paired_weight_bytes"] == self.plan["weights"]["paired_weight_bytes"],
                "Prepared rope/activation/all-output-pairs requirements not met")
        save(self.directory / "weight-cache-final.json", final)

    def dense(self):
        sys.path.insert(0, str(ROOT / "client"))
        from transport import decode
        for number, row in enumerate(self.plan["dense_prefix"]["layers"], 1):
            layer = row["layer"]
            _, values = decode((ROOT / row["frame"]).read_bytes())
            raw = np.asarray(values[27 * 2560 + 24576:], dtype="<f4").tobytes()
            path = self.directory / "inputs" / f"dense-{layer:02d}.f32"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            require(sha(path) == row["log_sha256"], "Dense prefix log changed")
            packet = BANKS[1][3] / f"layer-{layer:02d}.npf1"
            stats = self.call(f"dense-{layer:02d}", "prepare_fixed_prefix_state",
                              [{"file": str(path)}, {"file": str(packet)}],
                              idempotent=True, growth_bytes=2_097_152, did=DENSE_DID)
            require(stats[0] == number and stats[1] == number * 2_097_152,
                    "Unexpected dense prefix count/bytes; non-exclusive cache writer?")

    def prefixes(self):
        for bank in BANKS:  # common MUST be last; owner API cannot select a bank.
            name, _, _, packets = bank
            for layer in range(32):
                path = self.directory / "inputs" / f"{name}-{layer:02d}.f32"
                path.parent.mkdir(parents=True, exist_ok=True)
                values = bank_values(bank, layer)
                path.write_bytes(values.tobytes())
                packet = packets / f"layer-{layer:02d}.npf1"
                blob = {"file": str(packet)} if layer % 4 != 3 else {"hex": ""}
                self.call(f"prefix-{name}-{layer:02d}", "update_prefix", [layer, {"f32_file": str(path)}, blob],
                          growth_bytes=values.nbytes + (packet.stat().st_size if layer % 4 != 3 else 0))
        self.state["bootstrap_done"] = True
        save(self.path, self.state)

    def examples(self):
        require(self.state.get("bootstrap_done"), "Bootstrap must be checkpointed before owner inference")
        results = []
        for example in self.plan["examples"]:
            ident = example["id"]
            import time
            cost_path = self.directory / f"example-{ident}-cost.json"
            if cost_path.exists():
                cost = load(cost_path)
            else:
                existing_steps = sum(k.startswith(f"example-{ident}-") for k in self.state["calls"])
                cost = dict(before=self.cli.status(), started_unix=time.time(),
                            already_completed_steps_at_measurement_start=existing_steps)
                save(cost_path, cost)
            # Owner API embeds SUFFIX only; full IDs remain in preparation-plan provenance.
            progress = self.call(f"example-{ident}-start", "update_infer_start",
                                 [example["suffix_token_ids"], example["options"]])
            require(0 < progress["id"] and 0 < progress["stage"] <= 64,
                    "Invalid owner start progress")
            while True:
                require(progress["heap_pages"] * 65536 <= 4 * 1024**3 - self.args.heap_reserve_bytes,
                        "Owner heap exhausted configured 4GiB scratch safety margin")
                if progress["done"]:
                    require(progress["stage"] == 64 and progress["decision"] is not None,
                            "Completed owner graph lacks decision")
                    break
                stage = progress["stage"]
                job_id = progress["id"]
                next_progress = self.call(f"example-{ident}-stage-{stage:02d}", "update_infer_continue",
                                          [job_id, stage])
                require(next_progress["id"] == job_id and stage < next_progress["stage"] <= 64,
                        "Owner continuation did not advance expected job/stage")
                progress = next_progress
            from decision_validation import validate_decision
            decision = validate_decision(progress["decision"], example["options"])
            labels = example["options"] + ["__unknown__"]
            ui_result = dict(status="abstained" if decision["abstained"] else "answered",
                             value=decision["value"],
                             scores=dict(zip(labels, decision["probabilities"] +
                                             [decision["unknown_probability"]])),
                             raw_logits=dict(zip(labels, decision["raw_logits"])),
                             score_semantics="calibrated_normalized_scores",
                             calibration_version=decision["calibration_version"], reason=None)
            steps = [v for k, v in self.state["calls"].items() if k.startswith(f"example-{ident}-")]
            if "after" not in cost:
                cost["after"] = self.cli.status()
                integer = lambda x: int(str(x).replace("_", ""))
                cost["observed_canister_cycles_delta"] = integer(cost["before"]["cycles"]) - integer(cost["after"]["cycles"])
                cost["finished_unix"] = time.time()
                cost["wall_seconds"] = cost["finished_unix"] - cost["started_unix"]
                cost["scope"] = "Observed canister debit including owner inference, management/ingress overhead, idle rent, and any elapsed pause; not a public paid price."
            cost["graph_update_calls"] = len(steps)
            cost["reported_graph_instructions"] = sum(v["instructions"] for v in steps)
            cost["reported_stable_read_bytes"] = sum(v["stable_read_bytes"] for v in steps)
            cost["max_reported_heap_bytes"] = max(v["heap_pages"] * 65536 for v in steps)
            save(cost_path, cost)
            results.append(dict(id=ident, input=example["input"],
                                tokens=dict(total=len(example["token_ids"]), prefix=27,
                                            suffix=len(example["suffix_token_ids"])),
                                result=ui_result, owner_progress=progress, measured_cost=cost))
            # Actual canister outputs only. No native/gold expected choice assertion.
            save(self.directory / "owner-ui-results.json", dict(records=results))
        self.verify_ready()
        self.state["examples_done"] = True
        save(self.path, self.state)
        save(self.directory / "preparation-report.json",
             dict(bootstrap_done=True, examples_done=True, new_updates_this_run=self.updated,
                  plan_sha256=self.state["plan_sha256"], records=results,
                  paid_enabled=False, module_sha256=LIVE_GZIP_SHA))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="Explicitly authorize this runner's remote calls")
    ap.add_argument("--phase", choices=("all", "bootstrap", "examples"), default="all")
    ap.add_argument("--directory", default="artifacts/mainnet-deploy-20261007/preparation")
    ap.add_argument("--manifest", default="checkpoints/full-int8.manifest.json")
    ap.add_argument("--wasm", default="artifacts/paid-update-v1/build-v3/full.wasm")
    ap.add_argument("--prepared", default="artifacts/ui-extreme-examples-20261007/attempt2/prepared.json")
    ap.add_argument("--ui-results", default="artifacts/ui-extreme-examples-20261007/attempt2/results.json")
    ap.add_argument("--codec", default="target/release/examples/mainnet_candid")
    ap.add_argument("--floor-cycles", type=int, default=250_000_000_000)
    ap.add_argument("--max-new-updates", type=int, default=1000)
    ap.add_argument("--warm-concurrency", type=int, default=8,
                    help="Replay-safe warm calls per guarded batch (1..8); other updates stay serial")
    ap.add_argument("--heap-reserve-bytes", type=int, default=32 * 1024**2)
    args = ap.parse_args()
    require(args.floor_cycles >= 250_000_000_000, "Require at least freeze + 0.25T cycles")
    require(args.max_new_updates > 0 and 0 <= args.heap_reserve_bytes < 4 * 1024**3,
            "Invalid update cap/heap reserve")
    require(1 <= args.warm_concurrency <= 8, "Warm concurrency must be between 1 and 8")
    plan, tensors = offline_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    require((ROOT / args.codec).is_file(), "Missing offline Rust codec; parent must build mainnet_candid")
    runner = Runner(args, plan)
    runner.verify_ready()
    runner.weights(tensors)  # also validates full warmed cache before an examples-only resume
    if args.phase != "examples":
        runner.dense()
        runner.prefixes()
    if args.phase != "bootstrap":
        runner.examples()
    else:
        runner.verify_ready()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"STOP: {error}", file=sys.stderr)
        sys.exit(1)
