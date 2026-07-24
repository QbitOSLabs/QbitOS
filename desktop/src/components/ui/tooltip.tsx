import * as React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { cn } from "../../lib/utils";

/**
 * Tooltip provider — wrap the app root with this once.
 */
const TooltipProvider = TooltipPrimitive.Provider;

/**
 * Tooltip component built on Radix UI Tooltip primitive.
 * Enterprise blue color scheme with smooth fade-in.
 */
const Tooltip = TooltipPrimitive.Root;

const TooltipTrigger = TooltipPrimitive.Trigger;

const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Content
    ref={ref}
    sideOffset={sideOffset}
    className={cn(
      "z-50 overflow-hidden rounded-md bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] px-3 py-1.5 text-xs text-[var(--color-fg-muted)] shadow-md",
      "animate-in fade-in-0 zoom-in-95",
      className
    )}
    {...props}
  />
));
TooltipContent.displayName = "TooltipContent";

export { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider };
