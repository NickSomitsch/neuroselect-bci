import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import App from "./App";

describe("App", () => {
  it("states the project purpose and current limitation", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", { level: 1, name: "NeuroSelect" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Research boundary")).toHaveTextContent(
      "does not decode unrestricted thoughts",
    );
  });
});
