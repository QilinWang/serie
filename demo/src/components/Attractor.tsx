import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Bloom, EffectComposer } from "@react-three/postprocessing";
import { useMemo } from "react";
import type { DemoData } from "../lib/noise";
import { assembleNoisy, computeBounds } from "../lib/noise";
import { Trajectory } from "./Trajectory";

type AttractorProps = {
  data: DemoData;
  sigma: number;
  showNoisy: boolean;
  drawIndex: number;
};

export function Attractor({ data, sigma, showNoisy, drawIndex }: AttractorProps) {
  const bounds = useMemo(() => computeBounds(data.clean), [data.clean]);
  const draw = data.draws[drawIndex];
  const noisy = useMemo(
    () => (draw ? assembleNoisy(data.clean, draw, sigma) : null),
    [data.clean, draw, sigma],
  );

  return (
    <Canvas
      camera={{ position: [3, 2, 3], fov: 42, near: 0.01, far: 100 }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: false }}
    >
      <color attach="background" args={["#05070b"]} />
      <fog attach="fog" args={["#05070b", 4.2, 8.5]} />

      <group rotation={[-0.08, 0.18, 0]}>
        <Trajectory
          points={data.clean}
          center={bounds.center}
          scale={bounds.scale}
          hueRange={[210, 180]}
          opacity={0.98}
          lineWidth={2.1}
          blendMode="normal"
          lightness={0.52}
        />
        {showNoisy && sigma > 0 && noisy ? (
          <Trajectory
            points={noisy}
            colorSourcePoints={data.clean}
            center={bounds.center}
            scale={bounds.scale}
            hueRange={[18, 2]}
            opacity={0.34}
            lineWidth={1.9}
            blendMode="normal"
            lightness={0.43}
          />
        ) : null}
      </group>

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.055}
        autoRotate
        autoRotateSpeed={0.42}
        minDistance={1.2}
        maxDistance={8}
      />
      <EffectComposer>
        <Bloom intensity={0.24} luminanceThreshold={0.34} luminanceSmoothing={0.3} />
      </EffectComposer>
    </Canvas>
  );
}
