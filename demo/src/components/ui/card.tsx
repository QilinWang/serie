import * as React from "react";
import { cn } from "../../lib/utils";

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-md border border-border bg-black/55 shadow-panel backdrop-blur-xl",
        className,
      )}
      {...props}
    />
  );
}
