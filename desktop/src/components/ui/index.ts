/**
 * UI primitive components built on Radix UI.
 * Enterprise blue color scheme (no purple/pink/violet).
 *
 * Components follow the shadcn/ui pattern: each is a thin wrapper around
 * a Radix primitive with Tailwind styling that matches the QbitOS design
 * system (CSS custom properties defined in index.css).
 *
 * Available primitives:
 * - Avatar (with Image + Fallback)
 * - Button (with cva variants: default/secondary/ghost/danger/success/outline)
 * - Dialog (with Header/Footer/Title/Description)
 * - Label
 * - Progress
 * - ScrollArea (with custom ScrollBar)
 * - Separator
 * - Switch
 * - Tabs (with List/Trigger/Content)
 * - Toast (with Provider/Viewport/Title/Description/Close)
 * - Tooltip (with Provider/Trigger/Content)
 */

export { Avatar, AvatarImage, AvatarFallback } from "./avatar";
export { Button, buttonVariants, type ButtonProps } from "./button";
export {
  Dialog,
  DialogTrigger,
  DialogPortal,
  DialogClose,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
} from "./dialog";
export {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuCheckboxItem,
  DropdownMenuRadioItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuGroup,
  DropdownMenuPortal,
  DropdownMenuSub,
  DropdownMenuRadioGroup,
  DropdownMenuSubTrigger,
  DropdownMenuSubContent,
} from "./dropdown-menu";
export { Label } from "./label";
export { Progress } from "./progress";
export { ScrollArea, ScrollBar } from "./scroll-area";
export { Separator } from "./separator";
export { Switch } from "./switch";
export { Tabs, TabsList, TabsTrigger, TabsContent } from "./tabs";
export {
  ToastProvider,
  ToastViewport,
  Toast,
  ToastTitle,
  ToastDescription,
  ToastClose,
} from "./toast";
export {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  TooltipProvider,
} from "./tooltip";
