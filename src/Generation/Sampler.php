<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Generation;

/**
 * Token sampling strategies for text generation.
 *
 * Converts raw logits from the model into token IDs using
 * various sampling strategies (greedy, temperature, top-k, top-p).
 */
final class Sampler
{
    public function __construct(
        private readonly float $temperature = 1.0,
        private readonly int $topK = 0,
        private readonly float $topP = 1.0,
        private readonly float $repetitionPenalty = 1.0,
    ) {}

    /**
     * Sample a token ID from logits.
     *
     * @param float[] $logits Raw logits from the model (vocab_size length)
     * @param int[] $generatedIds Previously generated token IDs (for repetition penalty)
     */
    public function sample(array $logits, array $generatedIds = []): int
    {
        // Apply repetition penalty
        if ($this->repetitionPenalty !== 1.0 && !empty($generatedIds)) {
            foreach (array_unique($generatedIds) as $id) {
                if (isset($logits[$id])) {
                    if ($logits[$id] > 0) {
                        $logits[$id] /= $this->repetitionPenalty;
                    } else {
                        $logits[$id] *= $this->repetitionPenalty;
                    }
                }
            }
        }

        // Greedy: temperature 0 or very low
        if ($this->temperature <= 0.0001) {
            return $this->argmax($logits);
        }

        // Apply temperature
        if ($this->temperature !== 1.0) {
            foreach ($logits as &$logit) {
                $logit /= $this->temperature;
            }
            unset($logit);
        }

        // Apply top-k filtering
        if ($this->topK > 0) {
            $logits = $this->applyTopK($logits, $this->topK);
        }

        // Apply top-p (nucleus) filtering
        if ($this->topP < 1.0) {
            $logits = $this->applyTopP($logits, $this->topP);
        }

        // Convert to probabilities and sample
        $probs = $this->softmax($logits);
        return $this->weightedSample($probs);
    }

    /**
     * Return the index of the maximum value (greedy decoding).
     *
     * @param float[] $logits
     */
    private function argmax(array $logits): int
    {
        $maxVal = -INF;
        $maxIdx = 0;

        foreach ($logits as $i => $val) {
            if ($val > $maxVal) {
                $maxVal = $val;
                $maxIdx = $i;
            }
        }

        return $maxIdx;
    }

    /**
     * Keep only the top-K logits, setting everything else to -INF.
     *
     * @param float[] $logits
     * @return float[]
     */
    private function applyTopK(array $logits, int $k): array
    {
        $k = min($k, count($logits));
        $sorted = $logits;
        arsort($sorted);
        $topIndices = array_slice(array_keys($sorted), 0, $k, true);
        $topSet = array_flip($topIndices);

        foreach ($logits as $i => &$val) {
            if (!isset($topSet[$i])) {
                $val = -INF;
            }
        }
        unset($val);

        return $logits;
    }

    /**
     * Keep only the smallest set of tokens whose cumulative probability exceeds p.
     *
     * @param float[] $logits
     * @return float[]
     */
    private function applyTopP(array $logits, float $p): array
    {
        $probs = $this->softmax($logits);
        arsort($probs);

        $cumulative = 0.0;
        $keepSet = [];

        foreach ($probs as $i => $prob) {
            $cumulative += $prob;
            $keepSet[$i] = true;
            if ($cumulative >= $p) {
                break;
            }
        }

        foreach ($logits as $i => &$val) {
            if (!isset($keepSet[$i])) {
                $val = -INF;
            }
        }
        unset($val);

        return $logits;
    }

    /**
     * Compute softmax probabilities from logits.
     *
     * @param float[] $logits
     * @return float[]
     */
    private function softmax(array $logits): array
    {
        // Numerical stability: subtract max
        $max = max($logits);
        $exps = [];
        $sum = 0.0;

        foreach ($logits as $i => $val) {
            if ($val === -INF) {
                $exps[$i] = 0.0;
            } else {
                $exp = exp($val - $max);
                $exps[$i] = $exp;
                $sum += $exp;
            }
        }

        if ($sum > 0) {
            foreach ($exps as &$val) {
                $val /= $sum;
            }
            unset($val);
        }

        return $exps;
    }

    /**
     * Sample from a probability distribution.
     *
     * @param float[] $probs
     */
    private function weightedSample(array $probs): int
    {
        $r = mt_rand() / mt_getrandmax();
        $cumulative = 0.0;

        foreach ($probs as $i => $prob) {
            $cumulative += $prob;
            if ($r <= $cumulative) {
                return $i;
            }
        }

        // Fallback: return last token (shouldn't normally reach here)
        $keys = array_keys($probs);
        return end($keys);
    }

    /**
     * Create a greedy sampler (always picks the most likely token).
     */
    public static function greedy(): self
    {
        return new self(temperature: 0.0);
    }

    /**
     * Create a sampler with typical conversational settings.
     */
    public static function creative(float $temperature = 0.7, float $topP = 0.9): self
    {
        return new self(temperature: $temperature, topP: $topP);
    }
}
