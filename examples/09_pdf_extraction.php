<?php

/**
 * PDF Data Extraction with IBM Granite via ONNX Runtime.
 *
 * Demonstrates extracting structured data from PDFs using:
 * - spatie/pdf-to-text for PDF text extraction
 * - IBM Granite LLM (via ONNX Runtime) for intelligent data extraction
 *
 * Prerequisites:
 *   1. Install pdftotext: brew install poppler  (macOS)
 *   2. composer require spatie/pdf-to-text
 *   3. Download Granite model (see 08_text_generation.php)
 *
 * Usage:
 *   php examples/09_pdf_extraction.php path/to/invoice.pdf
 */

declare(strict_types=1);

require_once dirname(__DIR__) . '/vendor/autoload.php';

use PhpMlKit\ONNXRuntime\Generation\Sampler;
use PhpMlKit\ONNXRuntime\Generation\TextGenerator;

// Check for PDF argument
$pdfPath = $argv[1] ?? null;
if (!$pdfPath || !file_exists($pdfPath)) {
    echo "Usage: php examples/09_pdf_extraction.php <path-to-pdf>\n";
    exit(1);
}

// Check for spatie/pdf-to-text
if (!class_exists(\Spatie\PdfToText\Pdf::class)) {
    echo "Install spatie/pdf-to-text first:\n";
    echo "  composer require spatie/pdf-to-text\n";
    exit(1);
}

// Extract text from PDF
echo "Extracting text from PDF...\n";
$pdfText = (new \Spatie\PdfToText\Pdf())
    ->setPdf($pdfPath)
    ->text();

if (empty(trim($pdfText))) {
    echo "No text extracted from PDF. Is pdftotext installed? (brew install poppler)\n";
    exit(1);
}

echo "Extracted " . strlen($pdfText) . " characters from PDF.\n\n";

// Truncate if too long (LLM context limits)
$maxChars = 2000;
if (strlen($pdfText) > $maxChars) {
    $pdfText = substr($pdfText, 0, $maxChars) . "\n[... truncated ...]";
    echo "Note: Text truncated to {$maxChars} characters for inference.\n\n";
}

// Load the LLM
$modelDir = dirname(__DIR__) . '/models/granite-3.0-2b-instruct';

echo "Loading Granite 3.0 2B Instruct...\n";
$generator = TextGenerator::fromFiles(
    modelPath: "{$modelDir}/model_bnb4.onnx",
    tokenizerPath: "{$modelDir}/tokenizer.json",
);

// Build the extraction prompt
$prompt = $generator->formatChat([
    ['role' => 'system', 'content' => 'You are a data extraction assistant. Extract structured information from documents. Output valid JSON only, no explanation.'],
    ['role' => 'user', 'content' => "Extract the following fields from this document as JSON: company_name, date, total_amount, currency, line_items (array with description, quantity, unit_price).\n\nDocument text:\n---\n{$pdfText}\n---\n\nJSON:"],
]);

echo "Generating extraction...\n\n";

$result = $generator->generate(
    prompt: $prompt,
    maxTokens: 512,
    sampler: Sampler::greedy(),
    onToken: function (string $token) {
        echo $token;
        return true;
    },
);

echo "\n";
