import { describe, expect, it } from "vitest";
import { repairUnpairedSurrogates } from "../scripts/embedding-compat.js";

describe("embedding input Unicode compatibility", () => {
  it("retains valid supplementary characters", () => {
    expect(repairUnpairedSurrogates("before 🏆 after")).toEqual({ text: "before 🏆 after", replacedCodeUnits: 0 });
  });

  it("replaces only unpaired surrogate code units", () => {
    expect(repairUnpairedSurrogates(`left\ud83d middle\udc00 right`)).toEqual({
      text: "left� middle� right",
      replacedCodeUnits: 2,
    });
  });
});
