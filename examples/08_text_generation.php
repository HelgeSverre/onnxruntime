<?php

/**
 * Text Generation with IBM Granite via ONNX Runtime.
 *
 * This example demonstrates running an LLM entirely in PHP using
 * ONNX Runtime FFI bindings — no Python, no external APIs, fully offline.
 *
 * Prerequisites:
 *   1. Download the model and tokenizer:
 *      mkdir -p models/granite-3.0-2b-instruct
 *      cd models/granite-3.0-2b-instruct
 *      # Download from: https://huggingface.co/onnx-community/granite-3.0-2b-instruct
 *      # Files needed: model_bnb4.onnx, tokenizer.json, config.json
 *
 *   2. Run: php examples/08_text_generation.php
 */

declare(strict_types=1);

require_once dirname(__DIR__) . '/vendor/autoload.php';

use PhpMlKit\ONNXRuntime\Generation\Sampler;
use PhpMlKit\ONNXRuntime\Generation\TextGenerator;

$modelDir = dirname(__DIR__) . '/models/granite-3.0-2b-instruct';

// Check files exist
foreach (['model_bnb4.onnx', 'tokenizer.json', 'config.json'] as $file) {
    if (!file_exists("{$modelDir}/{$file}")) {
        echo "Missing: {$modelDir}/{$file}\n";
        echo "Download the Granite model first. See instructions above.\n";
        exit(1);
    }
}

echo "Loading Granite 3.0 2B Instruct...\n";
$t = microtime(true);

$generator = TextGenerator::fromFiles(
    modelPath: "{$modelDir}/model_bnb4.onnx",
    tokenizerPath: "{$modelDir}/tokenizer.json",
);

echo "Model loaded in " . round(microtime(true) - $t, 1) . "s\n\n";

// Format as a chat prompt
$prompt = $generator->formatChat([
    ['role' => 'system', 'content' => 'You are a helpful assistant. Be concise.'],
    ['role' => 'user', 'content' => 'What is PHP? Answer in one sentence.'],
]);

echo "Generating...\n\n";
echo "Assistant: ";

$t = microtime(true);
$tokenCount = 0;

$result = $generator->generate(
    prompt: $prompt,
    maxTokens: 100,
    sampler: Sampler::greedy(),
    onToken: function (string $token, int $tokenId) use (&$tokenCount) {
        echo $token;
        $tokenCount++;
        return true; // Continue generating
    },
);

$elapsed = microtime(true) - $t;
echo "\n\n";
echo "---\n";
echo "Generated {$tokenCount} tokens in " . round($elapsed, 1) . "s ";
echo "(" . round($tokenCount / $elapsed, 1) . " tokens/sec)\n";
