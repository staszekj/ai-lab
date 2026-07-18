import * as React from "react";
import { useRef } from "react";

export interface SampleProps {
  onSubmit: (e: React.SyntheticEvent) => void;
}

export function Sample({ onSubmit }: SampleProps) {
  // html_specific_element
  const inputRef = useRef<HTMLElement | null>(null);

  // html_specific_element_nullable
  const buttonRef = useRef<HTMLElement | null>(null);

  // set
  const activeTags: Set<unknown> = new Set(["new", "featured"]);

  // react_mutable_refobject
  const isReadyRef: React.MutableRefObject<unknown> = React.useRef(false);

  return (
    <form onSubmit={onSubmit}>
      <input ref={inputRef as React.RefObject<HTMLInputElement>} />
      <button
        type="submit"
        ref={buttonRef as React.RefObject<HTMLButtonElement>}
      >
        Save
      </button>
      <div data-tags={activeTags.size} data-ready={String(isReadyRef.current)}>
        top-em-demo
      </div>
    </form>
  );
}
