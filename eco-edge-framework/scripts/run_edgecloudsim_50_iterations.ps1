# Run EdgeCloudSim sample_app1 for 50 iterations (includes DQN_FIT policy).
# Requires: JDK, Maven/Gradle build of EdgeCloudSim JAR (see EdgeCloudSim README).

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$edgeSim = Join-Path $root "EdgeCloudSim"
$config = "scripts/sample_app1/config/default_config.properties"
$apps = "scripts/sample_app1/config/applications.xml"
$devices = "scripts/sample_app1/config/edge_devices.xml"

Push-Location $edgeSim
try {
    for ($i = 1; $i -le 50; $i++) {
        $out = "sim_results/ite$i"
        Write-Host "=== Iteration $i / 50 ==="
        java -cp "target/EdgeCloudSim.jar;lib/*" `
            edu.boun.edgecloudsim.applications.sample_app1.MainApp `
            $config $devices $apps $out $i
    }
    Write-Host "Done. Parse logs with: python ../parse_and_plot.py"
}
finally {
    Pop-Location
}
