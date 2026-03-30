<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Generation;

use PhpMlKit\ONNXRuntime\Enums\DataType;
use PhpMlKit\ONNXRuntime\Exceptions\InvalidArgumentException;
use PhpMlKit\ONNXRuntime\InferenceSession;
use PhpMlKit\ONNXRuntime\OrtValue;
use PhpMlKit\ONNXRuntime\SessionOptions;

/**
 * Autoregressive text generation using ONNX Runtime.
 *
 * Implements token-by-token generation with KV-cache management
 * for decoder-only transformer models (GPT-2, Granite, LLaMA, etc.).
 *
 * @example
 * $generator = TextGenerator::fromFiles(
 *     modelPath: 'models/granite/model_q4.onnx',
 *     tokenizerPath: 'models/granite/tokenizer.json',
 * );
 *
 * $text = $generator->generate('Hello, world!', maxTokens: 100);
 */
final class TextGenerator
{
    private InferenceSession $session;
    private BpeTokenizer $tokenizer;
    private int $numLayers;
    private int $numKvHeads;
    private int $headDim;
    private int $vocabSize;
    private DataType $kvDataType;
    private float $logitsScaling;

    /** @var string[] Names of KV-cache input tensors */
    private array $kvInputNames = [];

    /** @var string[] Names of KV-cache output tensors */
    private array $kvOutputNames = [];

    /** @var bool Whether the model expects position_ids input */
    private bool $hasPositionIds;

    private function __construct(
        InferenceSession $session,
        BpeTokenizer $tokenizer,
        array $modelConfig,
    ) {
        $this->session = $session;
        $this->tokenizer = $tokenizer;

        $this->numLayers = $modelConfig['num_hidden_layers'];
        $this->numKvHeads = $modelConfig['num_key_value_heads'];
        $this->headDim = (int) ($modelConfig['hidden_size'] / $modelConfig['num_attention_heads']);
        $this->vocabSize = $modelConfig['vocab_size'];
        $this->logitsScaling = $modelConfig['logits_scaling'] ?? 1.0;

        // Detect KV-cache data type from model outputs
        $this->kvDataType = $this->detectKvDataType();

        // Detect if model expects position_ids
        $this->hasPositionIds = isset($this->session->inputs()['position_ids']);

        // Cache the KV tensor names
        for ($i = 0; $i < $this->numLayers; ++$i) {
            $this->kvInputNames[] = "past_key_values.{$i}.key";
            $this->kvInputNames[] = "past_key_values.{$i}.value";
            $this->kvOutputNames[] = "present.{$i}.key";
            $this->kvOutputNames[] = "present.{$i}.value";
        }
    }

    /**
     * Create a text generator from model and tokenizer files.
     */
    public static function fromFiles(
        string $modelPath,
        string $tokenizerPath,
        ?string $configPath = null,
        ?SessionOptions $sessionOptions = null,
    ): self {
        $configPath ??= \dirname($modelPath).'/config.json';

        if (!file_exists($configPath)) {
            throw new InvalidArgumentException("Config file not found: {$configPath}");
        }

        $config = json_decode(file_get_contents($configPath), true, 512, \JSON_THROW_ON_ERROR);
        $tokenizer = BpeTokenizer::fromFile($tokenizerPath);
        $session = InferenceSession::fromFile($modelPath, $sessionOptions);

        return new self($session, $tokenizer, $config);
    }

    /**
     * Generate text from a prompt.
     *
     * @param string        $prompt    The input text
     * @param int           $maxTokens Maximum number of tokens to generate
     * @param null|Sampler  $sampler   Sampling strategy (defaults to greedy)
     * @param null|callable $onToken   callback for each generated token: fn(string $token, int $tokenId): bool
     *                                 Return false to stop generation early
     *
     * @return string The generated text (not including the prompt)
     */
    public function generate(
        string $prompt,
        int $maxTokens = 128,
        ?Sampler $sampler = null,
        ?callable $onToken = null,
    ): string {
        $sampler ??= Sampler::greedy();

        $inputIds = $this->tokenizer->encode($prompt);

        if (empty($inputIds)) {
            return '';
        }

        $generatedIds = [];
        $eosTokenId = $this->tokenizer->specialTokenId('<|end_of_text|>') ?? 0;

        // KV-cache state: starts empty, grows with each token
        $kvCache = null;
        $pastSeqLen = 0;

        for ($step = 0; $step < $maxTokens; ++$step) {
            if (0 === $step) {
                // Prefill: process all prompt tokens at once
                $currentIds = $inputIds;
            } else {
                // Decode: process only the last generated token
                $currentIds = [end($generatedIds)];
            }

            $seqLen = \count($currentIds);
            $totalSeqLen = $pastSeqLen + $seqLen;

            // Build inputs (creates new OrtValue objects for input_ids, attention_mask, position_ids)
            $inputs = $this->buildInputs($currentIds, $totalSeqLen, $kvCache, $pastSeqLen);

            // Request only logits and present KV-cache
            $outputNames = array_merge(['logits'], $this->kvOutputNames);

            // Run inference
            $outputs = $this->session->run($inputs, $outputNames);

            // Dispose input OrtValues we created (not KV-cache — those are output references)
            $inputs['input_ids']->dispose();
            $inputs['attention_mask']->dispose();
            if (isset($inputs['position_ids'])) {
                $inputs['position_ids']->dispose();
            }

            // Extract logits for the last token position
            // Shape: [1, seq_len, vocab_size] → we want [vocab_size] at the last position
            $logitsValue = $outputs['logits'];
            $allLogits = $logitsValue->toArray();
            $lastLogits = $allLogits[0][$seqLen - 1];

            // Apply Granite logits scaling
            if (1.0 !== $this->logitsScaling) {
                foreach ($lastLogits as &$logit) {
                    $logit /= $this->logitsScaling;
                }
                unset($logit);
            }

            // Sample next token
            $nextTokenId = $sampler->sample($lastLogits, array_merge($inputIds, $generatedIds));

            // Dispose logits output (no longer needed after sampling)
            $logitsValue->dispose();

            // Check for EOS
            if ($nextTokenId === $eosTokenId) {
                // Dispose remaining outputs before breaking
                foreach ($this->kvOutputNames as $name) {
                    if (isset($outputs[$name])) {
                        $outputs[$name]->dispose();
                    }
                }

                break;
            }

            $generatedIds[] = $nextTokenId;

            // Callback
            if (null !== $onToken) {
                $tokenText = $this->tokenizer->decode([$nextTokenId]);
                if (false === $onToken($tokenText, $nextTokenId)) {
                    // Dispose remaining outputs before breaking
                    foreach ($this->kvOutputNames as $name) {
                        if (isset($outputs[$name])) {
                            $outputs[$name]->dispose();
                        }
                    }

                    break;
                }
            }

            // Dispose old KV-cache from previous iteration before replacing
            if (null !== $kvCache) {
                foreach ($kvCache as $value) {
                    $value->dispose();
                }
            }

            // Update KV-cache for next step
            $kvCache = $this->extractKvCache($outputs);
            $pastSeqLen = $totalSeqLen;
        }

        // Dispose any remaining KV-cache after generation completes
        if (null !== $kvCache) {
            foreach ($kvCache as $value) {
                $value->dispose();
            }
        }

        return $this->tokenizer->decode($generatedIds);
    }

    /**
     * Format a prompt using the Granite chat template.
     *
     * @param array<array{role: string, content: string}> $messages
     */
    public function formatChat(array $messages): string
    {
        $prompt = '';
        foreach ($messages as $message) {
            $role = $message['role'];
            $content = $message['content'];
            $prompt .= "<|start_of_role|>{$role}<|end_of_role|>{$content}<|end_of_text|>\n";
        }
        // Add the assistant prefix to trigger generation
        $prompt .= '<|start_of_role|>assistant<|end_of_role|>';

        return $prompt;
    }

    /**
     * Get the tokenizer instance (for testing/debugging).
     */
    public function tokenizer(): BpeTokenizer
    {
        return $this->tokenizer;
    }

    /**
     * Build the input tensors for a single inference step.
     *
     * @param int[]                        $tokenIds    Current token IDs to process
     * @param int                          $totalSeqLen Total sequence length including past
     * @param null|array<string, OrtValue> $kvCache     Previous KV-cache tensors
     * @param int                          $pastSeqLen  Length of the past sequence (KV-cache)
     *
     * @return array<string, OrtValue>
     */
    private function buildInputs(
        array $tokenIds,
        int $totalSeqLen,
        ?array $kvCache,
        int $pastSeqLen,
    ): array {
        $seqLen = \count($tokenIds);

        // input_ids: [1, seq_len]
        $inputs = [
            'input_ids' => OrtValue::fromArray([$tokenIds], DataType::INT64, [1, $seqLen]),
        ];

        // attention_mask: [1, total_seq_len] — all ones
        $attentionMask = array_fill(0, $totalSeqLen, 1);
        $inputs['attention_mask'] = OrtValue::fromArray([$attentionMask], DataType::INT64, [1, $totalSeqLen]);

        // position_ids: [1, seq_len] — positions starting from pastSeqLen
        if ($this->hasPositionIds) {
            $positionIds = range($pastSeqLen, $pastSeqLen + $seqLen - 1);
            $inputs['position_ids'] = OrtValue::fromArray([$positionIds], DataType::INT64, [1, $seqLen]);
        }

        // KV-cache tensors
        if (null !== $kvCache) {
            // Use the KV-cache from previous step
            foreach ($this->kvInputNames as $name) {
                $outputName = str_replace('past_key_values.', 'present.', $name);
                if (isset($kvCache[$outputName])) {
                    $inputs[$name] = $kvCache[$outputName];
                }
            }
        } else {
            // First step: create empty KV-cache tensors
            // Shape: [1, num_kv_heads, 0, head_dim]
            $this->createEmptyKvCache($inputs);
        }

        return $inputs;
    }

    /**
     * Create empty KV-cache tensors for the first step.
     *
     * @param array<string, OrtValue> &$inputs
     */
    private function createEmptyKvCache(array &$inputs): void
    {
        foreach ($this->kvInputNames as $name) {
            // Create a zero-length tensor with shape [1, num_kv_heads, 0, head_dim]
            $shape = [1, $this->numKvHeads, 0, $this->headDim];
            $inputs[$name] = OrtValue::fromArray([], $this->kvDataType, $shape);
        }
    }

    /**
     * Extract KV-cache OrtValues from model outputs.
     *
     * @param array<string, OrtValue> $outputs
     *
     * @return array<string, OrtValue>
     */
    private function extractKvCache(array $outputs): array
    {
        $kvCache = [];
        foreach ($this->kvOutputNames as $name) {
            if (isset($outputs[$name])) {
                $kvCache[$name] = $outputs[$name];
            }
        }

        return $kvCache;
    }

    /**
     * Detect the data type used for KV-cache tensors from model metadata.
     */
    private function detectKvDataType(): DataType
    {
        $outputs = $this->session->outputs();

        // Check the first KV output tensor type
        if (isset($outputs['present.0.key'])) {
            return $outputs['present.0.key']['dtype'];
        }

        // Fallback to float32
        return DataType::FLOAT;
    }
}
