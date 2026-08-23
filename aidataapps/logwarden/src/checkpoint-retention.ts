export interface CheckpointReceiptIdentity {
  name: string;
  receiptSha256: string;
}

export function retainedCheckpointReceiptNames(
  receipts: CheckpointReceiptIdentity[],
  pinnedReceiptSha256: ReadonlySet<string>,
  keepLatest: number,
): Set<string> {
  if (!Number.isSafeInteger(keepLatest) || keepLatest < 1) throw new Error("keepLatest must be a positive integer");
  const ordered = [...receipts].sort((left, right) => left.name.localeCompare(right.name));
  return new Set(ordered
    .filter((receipt, index) => pinnedReceiptSha256.has(receipt.receiptSha256) || index >= ordered.length - keepLatest)
    .map((receipt) => receipt.name));
}
