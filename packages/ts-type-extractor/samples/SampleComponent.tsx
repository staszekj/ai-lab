/**
 * Sample React component with various TypeScript type annotations.
 * Used to test the type extractor.
 */
import React, { useState, useRef, useCallback, useEffect } from "react";

// ── Interface with typed properties ──────────────────────────────
interface ButtonProps {
  label: string;
  onClick: (e: React.MouseEvent<HTMLButtonElement>) => void;
  disabled?: boolean;
  variant: "primary" | "secondary" | "danger";
  icon?: React.ReactNode;
}

// ── Type alias ───────────────────────────────────────────────────
type FormState = {
  username: string;
  email: string;
  age: number;
};

// ── Functional component with explicit return type ───────────────
const Button: React.FC<ButtonProps> = ({ label, onClick, disabled, variant }): JSX.Element => {
  const buttonRef: React.RefObject<HTMLButtonElement> = useRef<HTMLButtonElement>(null);

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>): void => {
    console.log("clicked");
    onClick(e);
  };

  return (
    <button
      ref={buttonRef}
      onClick={handleClick}
      disabled={disabled}
      className={`btn btn-${variant}`}
    >
      {label}
    </button>
  );
};

// ── Component with useState generic types ────────────────────────
function LoginForm(): JSX.Element {
  const [formState, setFormState] = useState<FormState>({
    username: "",
    email: "",
    age: 0,
  });
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const inputRef = useRef<HTMLInputElement>(null);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>): void => {
      const { name, value }: { name: string; value: string } = e.target;
      setFormState((prev: FormState) => ({
        ...prev,
        [name]: value,
      }));
    },
    []
  );

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();
    setIsLoading(true);

    const response = await fetch("/api/login") as Response;
    const data: { token: string; user: { id: number; name: string } } = await response.json();

    setIsLoading(false);
  };

  useEffect((): (() => void) => {
    const timer: ReturnType<typeof setTimeout> = setTimeout(() => {
      inputRef.current?.focus();
    }, 100);

    return (): void => {
      clearTimeout(timer);
    };
  }, []);

  return (
    <form onSubmit={handleSubmit}>
      <input
        ref={inputRef}
        name="username"
        value={formState.username}
        onChange={handleChange}
      />
      <input
        name="email"
        type="email"
        value={formState.email}
        onChange={handleChange}
      />
      {error && <p className="error">{error}</p>}
      <Button
        label={isLoading ? "Loading..." : "Submit"}
        onClick={() => {}}
        variant="primary"
      />
    </form>
  );
}

// ── Event handler types ──────────────────────────────────────────
function EventHandlerExamples(): JSX.Element {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>): void => {
    const key: string = e.key;
    const code: number = e.keyCode;
  };

  const handleDrag = (e: React.DragEvent<HTMLDivElement>): void => {
    const data: DataTransfer = e.dataTransfer;
  };

  const handleFocus = (e: React.FocusEvent<HTMLInputElement>): void => {
    const target = e.target as HTMLInputElement;
  };

  return (
    <div onDragOver={handleDrag}>
      <input onKeyDown={handleKeyDown} onFocus={handleFocus} />
    </div>
  );
}

export { Button, LoginForm, EventHandlerExamples };
export type { ButtonProps, FormState };
