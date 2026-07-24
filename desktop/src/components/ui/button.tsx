import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { Slot } from "@radix-ui/react-slot";
import { cn } from "../../lib/utils";

/**
 * Button variants using class-variance-authority for type-safe variant composition.
 */
const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] disabled:pointer-events-none disabled:opacity-40",
  {
    variants: {
      variant: {
        default:
          "bg-[var(--color-accent)] text-white hover:shadow-[var(--shadow-glow)]",
        secondary:
          "bg-[var(--color-bg-elev-2)] text-[var(--color-fg)] border border-[var(--color-border)] hover:border-[var(--color-accent)]",
        ghost:
          "bg-transparent text-[var(--color-fg-muted)] hover:bg-[var(--color-bg-elev-2)] hover:text-[var(--color-fg)]",
        danger:
          "bg-[var(--color-danger)] text-white hover:opacity-90",
        success:
          "bg-[var(--color-success)] text-white hover:opacity-90",
        outline:
          "border border-[var(--color-border)] bg-transparent hover:bg-[var(--color-bg-elev-2)]",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-11 rounded-lg px-8 text-base",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

/**
 * Enterprise-grade Button component with variants.
 * Uses Radix Slot for polymorphic rendering (asChild prop).
 */
const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
