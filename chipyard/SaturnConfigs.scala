package chipyard

import org.chipsalliance.cde.config.{Config}
import saturn.common.{VectorParams}

// Rocket-integrated configs
class MINV64D64RocketConfig extends Config(
  new saturn.rocket.WithRocketVectorUnit(64, 64, VectorParams.minParams) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.AbstractConfig)

class MINV128D64RocketConfig extends Config(
  new saturn.rocket.WithRocketVectorUnit(128, 64, VectorParams.minParams) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.AbstractConfig)

class MINV256D64RocketConfig extends Config(
  new saturn.rocket.WithRocketVectorUnit(256, 64, VectorParams.minParams) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.AbstractConfig)

class REFV128D128RocketConfig extends Config(
  new saturn.rocket.WithRocketVectorUnit(128, 128, VectorParams.refParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.AbstractConfig)

class REFV256D64RocketConfig extends Config(
  new saturn.rocket.WithRocketVectorUnit(256, 64, VectorParams.refParams) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.AbstractConfig)

class REFV256D128RocketConfig extends Config(
  new saturn.rocket.WithRocketVectorUnit(256, 128, VectorParams.refParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.AbstractConfig)

class REFV256D128M64RocketConfig extends Config(
  new saturn.rocket.WithRocketVectorUnit(256, 128, VectorParams.refParams, mLen = Some(64)) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.AbstractConfig)

class REFV512D128RocketConfig extends Config(
  new saturn.rocket.WithRocketVectorUnit(512, 128, VectorParams.refParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.AbstractConfig)

class REFV512D256RocketConfig extends Config(
  new saturn.rocket.WithRocketVectorUnit(512, 256, VectorParams.refParams) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.AbstractConfig)

class DMAV256D256RocketConfig extends Config(
  new saturn.rocket.WithRocketVectorUnit(256, 256, VectorParams.dmaParams) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.AbstractConfig)

class MXV256D128RocketConfig extends Config(
  new saturn.rocket.WithRocketVectorUnit(256, 128, VectorParams.mxParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.AbstractConfig)

// Shuttle-integrated configs
class GENV128D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(128, 128, VectorParams.genParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class REFV256D64ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 64, VectorParams.refParams) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class REFV256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.refParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class DSPV256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.dspParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithSGTCM(address=0x78000000, size=(8L << 10), banks=16) ++
  new shuttle.common.WithTCM ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class GENV256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.genParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class MXV256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.mxParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class P3109V256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.p3109Params) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class P3109FiniteV256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.p3109FiniteParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class P3109BlockV256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.p3109BlockParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class P3109BlockFiniteV256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.p3109BlockFiniteParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

// Precision sweep (exu/fp/P3109Sweep.scala), named after the binary8pP formats
// the units carry; for synthesis only. P3109Sweep34 is the pair on the sweep's
// generic reader, the baseline. P3109Sweep346 is the control for
// P3109Sweep3456: if cost follows the widest precision rather than how many
// formats there are, the two match.
class P3109Sweep34V256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.p3109SweepParams()) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class P3109Sweep234V256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.p3109SweepParams(2)) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class P3109Sweep345V256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.p3109SweepParams(5)) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class P3109Sweep3456V256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.p3109SweepParams(5, 6)) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class P3109Sweep34567V256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.p3109SweepParams(5, 6, 7)) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class P3109Sweep234567V256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.p3109SweepParams(2, 5, 6, 7)) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class P3109Sweep346V256D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.p3109SweepParams(6)) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class REFV512D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(512, 128, VectorParams.refParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class DSPV512D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(512, 128, VectorParams.dspParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class GENV512D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(512, 128, VectorParams.genParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class GENV512D256ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(512, 256, VectorParams.genParams) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new shuttle.common.WithShuttleTileBeatBytes(32) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class GENV1024D128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(1024, 128, VectorParams.genParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class REFV256D256ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(256, 256, VectorParams.refParams) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new shuttle.common.WithShuttleTileBeatBytes(32) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class REFV512D256M128ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(512, 256, VectorParams.refParams, mLen = Some(128)) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class REFV512D256ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(512, 256, VectorParams.refParams) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new shuttle.common.WithShuttleTileBeatBytes(32) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class REFV512D512ShuttleConfig extends Config(
  new saturn.shuttle.WithShuttleVectorUnit(512, 512, VectorParams.refParams) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new shuttle.common.WithShuttleTileBeatBytes(64) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)


// Cosim configs

class MINV128D64RocketCosimConfig extends Config(
  new chipyard.harness.WithCospike ++
  new chipyard.config.WithTraceIO ++
  new saturn.rocket.WithRocketVectorUnit(128, 64, VectorParams.minParams) ++
  new freechips.rocketchip.rocket.WithCease(false) ++
  new freechips.rocketchip.rocket.WithDebugROB ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.AbstractConfig)

class GENV256D128ShuttleCosimConfig extends Config(
  new chipyard.harness.WithCospike ++
  new chipyard.config.WithTraceIO ++
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.genParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleDebugROB ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class MXV256D128ShuttleCosimConfig extends Config(
  new chipyard.harness.WithCospike ++
  new chipyard.config.WithTraceIO ++
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.mxParams) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new shuttle.common.WithShuttleDebugROB ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class MXV512D256ShuttleCosimConfig extends Config(
  new chipyard.harness.WithCospike ++
  new chipyard.config.WithTraceIO ++
  new saturn.shuttle.WithShuttleVectorUnit(512, 256, VectorParams.mxParams) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new shuttle.common.WithShuttleDebugROB ++
  new shuttle.common.WithShuttleTileBeatBytes(32) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)
