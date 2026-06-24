import { Line } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";
import type { Point3 } from "../lib/noise";

type TrajectoryProps = {
  points: Point3[];
  center: Point3;
  scale: number;
  hueRange: [number, number];
  colorSourcePoints?: Point3[];
  opacity?: number;
  lineWidth?: number;
  blendMode?: "normal" | "additive";
  lightness?: number;
};

export function Trajectory({
  points,
  center,
  scale,
  hueRange,
  colorSourcePoints = points,
  opacity = 1,
  lineWidth = 2,
  blendMode = "normal",
  lightness = 0.52,
}: TrajectoryProps) {
  const { scaledPoints, colors } = useMemo(() => {
    const n = points.length;
    if (n < 2) {
      return { scaledPoints: [] as Point3[], colors: [] as [number, number, number][] };
    }

    const velocities = new Array<number>(n - 1);
    for (let i = 0; i < n - 1; i += 1) {
      const a = colorSourcePoints[i] ?? points[i];
      const b = colorSourcePoints[i + 1] ?? points[i + 1];
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const dz = b[2] - a[2];
      velocities[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    velocities.push(velocities[velocities.length - 1] ?? 0);

    const maxVelocity = Math.max(...velocities, 1e-9);
    const scaledPoints: Point3[] = new Array(n);
    const colors: [number, number, number][] = new Array(n);
    const color = new THREE.Color();

    for (let i = 0; i < n; i += 1) {
      scaledPoints[i] = [
        (points[i][0] - center[0]) * scale,
        (points[i][1] - center[1]) * scale,
        (points[i][2] - center[2]) * scale,
      ];

      const t = velocities[i] / maxVelocity;
      const hue = hueRange[0] + t * (hueRange[1] - hueRange[0]);
      color.setHSL(hue / 360, 0.9, lightness);
      colors[i] = [color.r, color.g, color.b];
    }

    return { scaledPoints, colors };
  }, [center, colorSourcePoints, hueRange, points, scale]);

  return (
    <Line
      points={scaledPoints}
      vertexColors={colors}
      lineWidth={lineWidth}
      transparent
      opacity={opacity}
      blending={blendMode === "additive" ? THREE.AdditiveBlending : THREE.NormalBlending}
      depthWrite={false}
    />
  );
}
