// t24_amo_mesh -- atomics under real contention, on all four tiles at once.
//
// t22_atomic checks the arithmetic of every operation on one core. This checks
// the property that only a multi-core machine can check: that a
// read-modify-write is INDIVISIBLE when four cores are doing it to the same
// address through the mesh.
//
// ISA sec 9.2 puts atomics at the home slice of the address, and the home is a
// pure function of the line address (sec 10.1), so all four tiles' requests to
// COUNTER queue at one slice and the slice's arbiter serialises them (sec
// 14.1). This program is what that claim looks like when it is true.
//
// HOW IT CHECKS ITSELF, with no help from the bench
//
// Each tile adds 1 to COUNTER exactly N times, then spins until it reads
// exactly 4*N. Equality, not >=, is the whole point:
//
//   - a lost update (two cores reading the same old value) leaves the total
//     below 4*N forever, and every tile spins until the bench times out;
//   - a double-counted update puts it above 4*N, and the equality spin never
//     matches, so that fails too.
//
// Reaching `exit` on all four tiles therefore means the total is exactly 4*N,
// which is the definition of atomic. The other two counters check that the
// non-additive operations serialise the same way: FLAGS must end up with one
// bit per tile, and MAXID with the largest tile id.
.org 0

.equ  COUNTER,  0x9000
.equ  FLAGS,    0x9040          // a different line, so a different home slice
.equ  MAXID,    0x9080
.equ  N_ADD,    64
.equ  N_TILES,  4

start:
    s.rdspecial s1, core_id            // 0..3
    s.rdspecial s23, role_id           // 0..NUM_ROLES-1
    s.rdspecial s24, num_roles

    // Every Role Context on every tile takes part, so the expected total is
    // tiles * roles * N_ADD and not a constant. The bench launches one role or
    // all of them (+allroles); a test that hard-coded four contenders would
    // hang in exactly the configuration that exercises the most contention.

    // ---- context 0 of tile 0 initialises, everyone else waits for it -------
    // The init flag is the counter line itself: every tile spins until it can
    // see a value, which for tiles 1..3 means tile 0's store has reached the
    // home slice. A store is complete at its home slice when the instruction
    // retires (sec 9.2), so no fence is needed to make it visible -- only to
    // order it against what this tile does next.
    .li      s2, #COUNTER
    .li      s3, #FLAGS
    .li      s4, #MAXID
    br.nz    s1, contend
    br.nz    s23, contend

    .li      s5, #0
    s.st.d   [s2, #0], s5
    s.st.d   [s3, #0], s5
    s.st.d   [s4, #0], s5
    fence.release.device               // publish the zeroes before anyone adds
    .li      s6, #1
    .li      s7, #0x9100
    s.st.d   [s7, #0], s6              // "initialised"
    br       contend_ready

contend:
    // Tiles 1..3 wait for the flag rather than assuming an order.
    .li      s7, #0x9100
wait_init:
    atom.ld.d.acquire s8, [s7]
    br.z     s8, wait_init

contend_ready:
    // ---- N_ADD atomic increments, from every tile at once ------------------
    .li      s9, #N_ADD
    .li      s10, #1
add_loop:
    atom.add.d.acq_rel.device s11, [s2], s10
    s.addi   s9, s9, #-1
    br.nz    s9, add_loop

    // ---- one bit per tile, and the largest id -----------------------------
    .li      s12, #1
    s.shl    s12, s12, s1               // 1 << core_id
    atom.or.d.acq_rel.device s13, [s3], s12
    atom.max.u.d.acq_rel.device s14, [s4], s1

    // ---- spin until the total is EXACTLY tiles*roles*N_ADD -----------------
    .li      s15, #N_TILES
    .li      s16, #N_ADD
    s.mul    s17, s15, s16
    s.mul    s17, s17, s24              // * the number of Role Contexts
wait_total:
    atom.ld.d.acquire s18, [s2]
    s.cmp.ne s19, s18, s17
    br.nz    s19, wait_total

    // ---- and until every tile has set its flag bit -------------------------
    .li      s20, #0x0f                 // (1<<4)-1
wait_flags:
    atom.ld.d.acquire s21, [s3]
    s.cmp.ne s19, s21, s20
    br.nz    s19, wait_flags

    // MAXID is settled once every tile has flagged, so this needs no spin.
    atom.ld.d.acquire s22, [s4]
    .assert_eq s22, #3, #1

    fence.acq_rel.device
    exit
