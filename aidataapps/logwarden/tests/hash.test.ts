import { canonicalJson, hashJson, sha256 } from "../src/hash.js";

describe("canonical hashing", () => {
  it("sorts object keys recursively without reordering arrays", () => {
    const left = { z: [{ b: 2, a: 1 }], a: true };
    const right = { a: true, z: [{ a: 1, b: 2 }] };
    expect(canonicalJson(left)).toBe('{"a":true,"z":[{"a":1,"b":2}]}');
    expect(hashJson(left)).toBe(hashJson(right));
    expect(sha256("abc")).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  });
});
