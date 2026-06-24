import { Card } from "./ui/card";
import { Slider } from "./ui/slider";
import { Switch } from "./ui/switch";

type ControlsProps = {
  sigma: number;
  showNoisy: boolean;
  system: string;
  seed: number;
  nPoints: number;
  nDraws: number;
  drawIndex: number;
  hasNoiseDraws: boolean;
  onSigmaChange: (sigma: number) => void;
  onShowNoisyChange: (show: boolean) => void;
  onDrawIndexChange: (index: number) => void;
};

export function Controls({
  sigma,
  showNoisy,
  system,
  seed,
  nPoints,
  nDraws,
  drawIndex,
  hasNoiseDraws,
  onSigmaChange,
  onShowNoisyChange,
  onDrawIndexChange,
}: ControlsProps) {
  const drawOptions = Array.from({ length: nDraws }, (_, index) => index);

  return (
    <Card className="w-full max-w-sm p-4">
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
          <dt className="text-slate-500">seed</dt>
          <dd className="mt-1 font-mono text-slate-100">{seed}</dd>
        </div>
        <div className="rounded-md bg-white/[0.04] p-2">
          <dt className="text-slate-500">points</dt>
          <dd className="mt-1 font-mono text-slate-100">{nPoints.toLocaleString()}</dd>
        </div>
        <div className="rounded-md bg-white/[0.04] p-2">
          <dt className="text-slate-500">draws</dt>
          <dd className="mt-1 font-mono text-slate-100">{nDraws}</dd>
        </div>
      </dl>
    </Card>
  );
}
