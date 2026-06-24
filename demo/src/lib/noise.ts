export type Point3 = [number, number, number];

export type DemoData = {
  system: string;
  seed: number;
  feature_names: string[];
  dt: number;
  n_points: number;
  clean: Point3[];
  draws: Point3[][];
};

export type Bounds = {
  center: Point3;
  scale: number;
};

export function computeBounds(points: Point3[]): Bounds {
  if (points.length === 0) {
    return { center: [0, 0, 0], scale: 1 };
  }

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;

  for (const [x, y, z] of points) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }

  const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-9);
  return {
    center: [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2],
    scale: 2.65 / span,
  };
}

export function assembleNoisy(clean: Point3[], draw: Point3[], sigma: number): Point3[] {
  const n = Math.min(clean.length, draw.length);
  const out: Point3[] = new Array(n);
  for (let i = 0; i < n; i += 1) {
    out[i] = [
      clean[i][0] + sigma * draw[i][0],
      clean[i][1] + sigma * draw[i][1],
      clean[i][2] + sigma * draw[i][2],
    ];
  }
  return out;
}
