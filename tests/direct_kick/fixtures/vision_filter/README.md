# DirectKick VisionFilter C++ numerical oracle

This golden fixture uses original `cvkf.cpp` and `select_best_prediction.cpp` from `futbol_main` main revision `32ece6ee0676b1008d5bc58c3533d45613440568`. Source files are compiled directly. No production files were changed; no ROS nodes are spun. The bank orchestration in `oracle.cpp` is a small copy of node logic at `vision_filter_node.cpp:632–787`, using original C++ core and selector classes.

`cpp_oracle.json` is the checked-in fixture; it records revision and SHA-256 hashes, including the driver. Regenerate with `python3 tests/direct_kick/fixtures/vision_filter/build.py --source-root /path/to/pinned/futbol_main`. Compilation and output go to `/tmp/direct_kick_vision_oracle` by default. The generated `fixtures.json` is equivalent with expanded whitespace; `compile_command.json` records compiler argv. Regeneration requires the existing ROS generated headers, Eigen, and source workspace `build/vision_filter/build.ninja` include metadata. Running the Python tests requires none of those C++ dependencies.

Run the simulator-independent tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/direct_kick/test_vision_filter.py`. Disabling automatic plugins avoids unrelated ROS test-plugin imports.

## Fixture groups

- `core_analytic`: initialization and 0.1 s missing observation for independent stationary and rolling CVKFs. Initial SI P diagonal is `[.0625,.0625,6.25,6.25]`. After missing: stationary Pxx=.125016, Pxvx=.00032, Pvxvx=.0064; rolling Pxx=.125016, Pxvx=.62532, Pvxvx=6.2564.
- `core`: same 17-input series independently fed to stationary and rolling CVKFs. Core directly accepts finite observations; it has no node NIS gating. Includes raw NIS returned before mutation and original innovation likelihood after correction. Group by `hypothesis` and instantiate/reset one corresponding core before replay.
- `bank`: 32 two-dimensional moving measurements with deterministic perturbations, followed by reversal. Every frame is passed to the bank as accepted; pre-step NIS is diagnostic. Replays MAP selection and post-selection bounce reseed. Actual selected hypotheses include stationary, rolling, bounce.
- `node`: complete confirmed/tentative orchestration. Covers capture1→promotion, confirmation, missing, far outlier→new coherent track promotion while old track still active, abandoned tentative when confirmed accepts again, exact 3.0 s predicted boundary and +1 ns LOST, interrupted reacquisition, transform-invalid input.
- `interleaved`: three independent node drivers, with records interleaved by environment. Env0 equals `node`; env1 moves continuously; env2 has distinct positions/ranges and periodic missing input. Replay in one batched implementation with active-env masks to check isolation.

## Mapping

Hypothesis indices: 0 stationary, 1 rolling, 2 high_speed, 3 bounce. Status: 0 LOST, 1 OBSERVED, 2 PREDICTED. All input/output state values are SI. Core arithmetic is original C++ double/cm. Input timestamps are integer nanoseconds; elapsed times must come from timestamps. Both global and local positions are explicit; local range controls R and differs from global range.

`output.state`: `[x,y,vx,vy]`. `covariance`: 16 row-major entries. Core internal LOST output has state=null but can retain covariance. Node public LOST snapshot has state/P zero and velocity_valid=false. Initial non-LOST output has velocity_valid=true even though estimated velocity is zero. `measurement_accepted` is exposed by this fixture for parity checks; it is not a field of the ROS public message.

`innovation_log_likelihood_cm`: original units. With an SI-unit Torch core, `loglike_cpp = loglike_SI - log(10000)`; posterior selection is unaffected by this common constant. `confidence_internal` precedes node detector-confidence override. The fixture does not synthesize detector confidence.

`future_xy_0_1` / `future_covariance_0_1` are the original core's internal 0.1 s forecast, included for verification only. These are not public VisionFilter fields and are not the user-approved 13-horizon public-snapshot adapter.

Bank trace: `nis` contains pre-step optional values; gate short-circuits on the first absent NIS, leaving later entries null. `hypotheses` contains all outputs before bounce reseed. `selected` is original C++ MAP index. `bounce_post_state` / `bounce_post_covariance` come from the actual bounce filter after potential reseed (its velocity reconstructed via future_position(.1)-future_position(0), so allow floating-point roundoff).

Node traces retain which bank ran before a promotion swap. `confirmed_active`, `capture_count` and `event` are post-call diagnostic values; they are not ROS message fields. Confirmed/tentative bank state, including model probabilities, moves together on promotion. `selected=-1` means that bank was not stepped.

The driver inputs are finite; finite-data checks are represented by observed/transform flags. It omits field boundaries, localization synchronization, raw detection selection and detector-confidence override. Those are node-adapter concerns beyond these numerical bank fixtures.

Suggested numeric comparisons: float64 state/P atol 1e-10 with a modest rtol, policy float32 after explicit conversion. Keep exact assertions for statuses, capture counts and acceptance. Identical model likelihoods and nearly tied posteriors can make MAP decisions sensitive to arithmetic evaluation order; inspect the per-hypothesis records if a cross-language comparison diverges rather than relaxing status/state behavior blindly.
