import { useEffect, useMemo, useState } from "react";
import { Attractor } from "./components/Attractor";
import { Controls } from "./components/Controls";
import type { DemoData } from "./lib/noise";

type LoadState =
  | { status: "loading" }
  | { status: "ready"; data: DemoData }
  | { status: "error"; message: string };

declare global {
  interface Window {
    __SERIE_DEMO_DATA__?: unknown;
  }
}

function isDemoData(value: unknown): value is DemoData {
  if (!value || typeof value !== "object") return false;
  const data = value as Partial<DemoData>;
  return (
    typeof data.system === "string" &&
    typeof data.seed === "number" &&
    Array.isArray(data.feature_names) &&
    Array.isArray(data.clean) &&
    Array.isArray(data.draws)
  );
}

export default function App() {
  const [loadState, setLoadState] = useState<LoadState>({ status: "loading" });
  const [sigma, setSigma] = useState(0.24);
  const [showNoisy, setShowNoisy] = useState(true);
  const [drawIndex, setDrawIndex] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const inlineData = window.__SERIE_DEMO_DATA__;
    if (isDemoData(inlineData)) {
      setLoadState({ status: "ready", data: inlineData });
      return () => {
        cancelled = true;
      };
    }

    fetch("attractor_data.json")
      .then((response) => {
        if (!response.ok) {
          throw new Error(`failed to load attractor_data.json (${response.status})`);
        }
        return response.json();
      })
      .then((payload: unknown) => {
        if (!isDemoData(payload)) {
          throw new Error("attractor_data.json does not match the SERIE demo contract");
        }
        if (!cancelled) {
          setLoadState({ status: "ready", data: payload });
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          const message = error instanceof Error ? error.message : String(error);
          setLoadState({ status: "error", message });
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const content = useMemo(() => {
    if (loadState.status === "loading") {
      return (
        <div className="grid h-screen place-items-center bg-background px-6 text-center text-slate-300">
          <div>
            <p className="text-xs uppercase tracking-[0.22em] text-cyan-200">SERIE</p>
            <p className="mt-3 text-lg">Loading baked attractor data...</p>
          </div>
        </div>
      );
    }

    if (loadState.status === "error") {
      return (
        <div className="grid h-screen place-items-center bg-background px-6 text-center text-slate-300">
          <div className="max-w-md rounded-md border border-red-300/20 bg-red-950/30 p-5">
            <p className="text-xs uppercase tracking-[0.22em] text-red-200">Missing demo data</p>
            <h1 className="mt-2 text-xl font-semibold text-white">Bake the SERIE JSON first.</h1>
            <p className="mt-3 text-sm leading-6 text-red-100/80">{loadState.message}</p>
            <code className="mt-4 block rounded-md bg-black/40 px-3 py-2 text-left text-xs text-red-50">
              bun run bake
            </code>
          </div>
        </div>
      );
    }

    const { data } = loadState;
    const effectiveDrawIndex =
      data.draws.length > 0 ? Math.min(drawIndex, data.draws.length - 1) : 0;
    return (
      <main className="relative h-screen w-screen overflow-hidden bg-background">
        <div className="absolute inset-0 sm:left-[26rem]">
          <Attractor
            data={data}
            sigma={sigma}
            showNoisy={showNoisy}
            drawIndex={effectiveDrawIndex}
          />
        </div>

        <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-center px-4 pt-4 sm:justify-start sm:p-6">
          <div className="pointer-events-auto w-full max-w-sm">
            <Controls
              sigma={sigma}
              showNoisy={showNoisy}
              system={data.system}
              seed={data.seed}
              nPoints={data.n_points}
              nDraws={data.draws.length}
              drawIndex={effectiveDrawIndex}
              hasNoiseDraws={data.draws.length > 0}
              onSigmaChange={setSigma}
              onShowNoisyChange={setShowNoisy}
              onDrawIndexChange={setDrawIndex}
            />
          </div>
        </div>

        <div className="pointer-events-none absolute bottom-4 left-4 right-4 flex flex-col gap-2 text-xs text-slate-400 sm:bottom-6 sm:left-[26rem] sm:right-6 sm:flex-row sm:items-end sm:justify-between">
          <p className="max-w-xl leading-5">
            Clean truth is blue. The orange path is assembled in the browser from the same
            selected frozen noise draw at the selected σ.
          </p>
          <p className="font-mono text-slate-500">
            {data.feature_names.slice(0, 3).join(" / ")} · dt={data.dt}
          </p>
        </div>
      </main>
    );
  }, [drawIndex, loadState, showNoisy, sigma]);

  return content;
}
