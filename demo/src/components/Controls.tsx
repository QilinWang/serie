import { Card } from "./ui/card";
import { Slider } from "./ui/slider";
import { Switch } from "./ui/switch";
import type { DemoData } from "../lib/noise";

type ControlsProps = {
  sigma: number;
  showNoisy: boolean;
  systems: DemoData[];
  system: string;
  seed: number;
  nPoints: number;
  nDraws: number;
  drawIndex: number;
  hasNoiseDraws: boolean;
  sourceDim?: number;
  projection?: string;
  shape?: DemoData["shape"];
  onSystemChange: (system: string) => void;
  onSigmaChange: (sigma: number) => void;
  onShowNoisyChange: (show: boolean) => void;
  onDrawIndexChange: (index: number) => void;
  onLoadDataFile: (file: File) => void;
};

export function Controls({
  sigma,
  showNoisy,
  systems,
  system,
  seed,
  nPoints,
  nDraws,
  drawIndex,
  hasNoiseDraws,
  sourceDim,
  projection,
  shape,
  onSystemChange,
  onSigmaChange,
  onShowNoisyChange,
  onDrawIndexChange,
  onLoadDataFile,
}: ControlsProps) {
  const drawOptions = Array.from({ length: nDraws }, (_, index) => index);
  const cleanShape = shape?.clean ?? [nPoints, 3];
  const drawShape = shape?.draws ?? [nDraws, nPoints, 3];

  return (
    <Card className="max-h-[calc(100vh-2rem)] w-full max-w-sm overflow-y-auto p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-cyan-200/80">SERIE</p>
          <h1 className="mt-1 text-xl font-semibold tracking-normal text-white">Known Truth, Dosed Noise</h1>
        </div>
        <div className="rounded-md border border-cyan-200/20 bg-cyan-200/10 px-2 py-1 text-sm tabular-nums text-cyan-100">
          σ={sigma.toFixed(2)}
        </div>
      </div>

      <p className="mt-3 text-sm leading-6 text-slate-300">
        Drag to explore the clean attractor. Move the slider to add observation
        noise from the baked SERIE draw bank in natural unit scale.
      </p>

      <div className="mt-5 rounded-md border border-white/10 bg-white/[0.04] px-3 py-2">
        <label htmlFor="system-select" className="text-sm font-medium text-slate-100">
          System
        </label>
        <select
          id="system-select"
          value={system}
          onChange={(event) => onSystemChange(event.currentTarget.value)}
          className="mt-2 h-9 w-full rounded-md border border-cyan-200/20 bg-black/40 px-2 text-sm font-mono text-cyan-100 outline-none focus:border-cyan-200"
        >
          {systems.map((item) => {
            const itemShape = item.shape?.clean ?? [item.n_points, item.source_dim ?? 3];
            return (
              <option key={item.system} value={item.system}>
                {item.system} · {itemShape[0]}x{itemShape[1]}
              </option>
            );
          })}
        </select>
      </div>

      <div className="mt-3 rounded-md border border-white/10 bg-white/[0.04] px-3 py-2">
        <label htmlFor="data-json" className="text-sm font-medium text-slate-100">
          Load local JSON
        </label>
        <p className="mt-1 text-xs leading-5 text-slate-400">
          Select a SERIE demo-data JSON file; it stays in this browser.
        </p>
        <input
          id="data-json"
          type="file"
          accept="application/json,.json"
          onChange={(event) => {
            const file = event.currentTarget.files?.[0];
            if (file) onLoadDataFile(file);
            event.currentTarget.value = "";
          }}
          className="mt-2 block w-full text-xs text-slate-300 file:mr-3 file:rounded-md file:border-0 file:bg-cyan-200/15 file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-cyan-100 hover:file:bg-cyan-200/25"
        />
      </div>

      <div className="mt-5 space-y-3">
        <div className="flex items-center justify-between text-sm">
          <label htmlFor="sigma-slider" className="font-medium text-slate-100">
            Observation noise
          </label>
          <span className="font-mono text-xs text-slate-300">clean + σ · ε</span>
        </div>
        <Slider
          id="sigma-slider"
          value={[sigma]}
          min={0}
          max={1}
          step={0.01}
          onValueChange={(value) => onSigmaChange(value[0] ?? 0)}
        />
        <p className="text-xs leading-5 text-slate-400">
          σ is the unit-noise multiplier: σ=1 applies one natural-scale draw from
          the declared observation-noise law.
        </p>
      </div>

      <div className="mt-5 flex items-center justify-between gap-3 rounded-md border border-white/10 bg-white/[0.04] px-3 py-2">
        <div>
          <label htmlFor="show-noisy" className="text-sm font-medium text-slate-100">
            Show noisy observation
          </label>
          <p className="text-xs text-slate-400">Scales the selected frozen ε draw.</p>
        </div>
        <Switch
          id="show-noisy"
          checked={showNoisy && hasNoiseDraws}
          disabled={!hasNoiseDraws}
          onCheckedChange={onShowNoisyChange}
        />
      </div>

      <div className="mt-3 flex items-center justify-between gap-3 rounded-md border border-white/10 bg-white/[0.04] px-3 py-2">
        <div>
          <label htmlFor="draw-index" className="text-sm font-medium text-slate-100">
            Noise realization
          </label>
          <p className="text-xs text-slate-400">
            Selects ε[index], the nth frozen realization reused across σ.
          </p>
        </div>
        <select
          id="draw-index"
          value={drawIndex}
          disabled={!hasNoiseDraws}
          onChange={(event) => onDrawIndexChange(Number(event.currentTarget.value))}
          className="h-8 rounded-md border border-cyan-200/20 bg-black/40 px-2 text-sm font-mono text-cyan-100 outline-none focus:border-cyan-200 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {drawOptions.map((index) => (
            <option key={index} value={index}>
              draw {index}
            </option>
          ))}
        </select>
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-2 text-xs text-slate-300">
        <div className="rounded-md bg-white/[0.04] p-2">
          <dt className="text-slate-500">source</dt>
          <dd className="mt-1 font-mono text-slate-100">{system}</dd>
        </div>
        <div className="rounded-md bg-white/[0.04] p-2">
          <dt className="text-slate-500">clean shape</dt>
          <dd className="mt-1 font-mono text-slate-100">
            {cleanShape[0]}x{cleanShape[1]}
          </dd>
        </div>
        <div className="rounded-md bg-white/[0.04] p-2">
          <dt className="text-slate-500">seed</dt>
          <dd className="mt-1 font-mono text-slate-100">{seed}</dd>
        </div>
        <div className="rounded-md bg-white/[0.04] p-2">
          <dt className="text-slate-500">points</dt>
          <dd className="mt-1 font-mono text-slate-100">{nPoints.toLocaleString()}</dd>
        </div>
        <div className="rounded-md bg-white/[0.04] p-2">
          <dt className="text-slate-500">draws</dt>
          <dd className="mt-1 font-mono text-slate-100">
            {drawShape[0]}x{drawShape[1]}x{drawShape[2]}
          </dd>
        </div>
        <div className="rounded-md bg-white/[0.04] p-2">
          <dt className="text-slate-500">source dim</dt>
          <dd className="mt-1 font-mono text-slate-100">{sourceDim ?? 3}</dd>
        </div>
        <div className="rounded-md bg-white/[0.04] p-2">
          <dt className="text-slate-500">projection</dt>
          <dd className="mt-1 truncate font-mono text-slate-100" title={projection ?? "native_3d"}>
            {projection ?? "native_3d"}
          </dd>
        </div>
      </dl>
    </Card>
  );
}
