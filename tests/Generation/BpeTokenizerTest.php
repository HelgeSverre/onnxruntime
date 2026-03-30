<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Tests\Generation;

use PhpMlKit\ONNXRuntime\Generation\BpeTokenizer;
use PHPUnit\Framework\TestCase;

/**
 * @internal
 *
 * @coversNothing
 */
class BpeTokenizerTest extends TestCase
{
    private static ?BpeTokenizer $tokenizer = null;

    public static function setUpBeforeClass(): void
    {
        $path = \dirname(__DIR__, 2).'/models/granite-3.0-2b-instruct/tokenizer.json';
        if (!file_exists($path)) {
            self::markTestSkipped('Granite tokenizer.json not found. Download it first.');
        }
        self::$tokenizer = BpeTokenizer::fromFile($path);
    }

    public function testVocabSize(): void
    {
        $this->assertGreaterThan(49000, self::$tokenizer->vocabSize());
    }

    public function testEncodeSimple(): void
    {
        $ids = self::$tokenizer->encode('Hello');
        $this->assertNotEmpty($ids);
        $this->assertContainsOnly('int', $ids);
    }

    public function testRoundTrip(): void
    {
        $texts = [
            'Hello, world!',
            'The quick brown fox jumps over the lazy dog.',
            'Extract: invoice #12345, total $99.99',
            "Multi\nline\ntext",
            'Unicode: café résumé naïve',
        ];

        foreach ($texts as $text) {
            $ids = self::$tokenizer->encode($text);
            $decoded = self::$tokenizer->decode($ids);
            $this->assertSame($text, $decoded, "Round-trip failed for: {$text}");
        }
    }

    public function testSpecialTokenIds(): void
    {
        $this->assertSame(0, self::$tokenizer->specialTokenId('<|end_of_text|>'));
        $this->assertSame(49152, self::$tokenizer->specialTokenId('<|start_of_role|>'));
        $this->assertSame(49153, self::$tokenizer->specialTokenId('<|end_of_role|>'));
    }

    public function testSpecialTokensInEncoding(): void
    {
        $text = '<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>';
        $ids = self::$tokenizer->encode($text);

        $this->assertContains(49152, $ids); // start_of_role
        $this->assertContains(49153, $ids); // end_of_role
        $this->assertContains(0, $ids);     // end_of_text
    }

    public function testDecodeSkipsSpecialTokensByDefault(): void
    {
        $ids = [49152, 496, 49153]; // <|start_of_role|> user <|end_of_role|>
        $decoded = self::$tokenizer->decode($ids, skipSpecialTokens: true);
        $this->assertSame('user', $decoded);
    }

    public function testDecodeWithSpecialTokens(): void
    {
        $ids = [49152, 496, 49153]; // <|start_of_role|> user <|end_of_role|>
        $decoded = self::$tokenizer->decode($ids, skipSpecialTokens: false);
        $this->assertSame('<|start_of_role|>user<|end_of_role|>', $decoded);
    }

    public function testEmptyInput(): void
    {
        $this->assertSame([], self::$tokenizer->encode(''));
    }

    public function testChatTemplateRoundTrip(): void
    {
        $prompt = "<|start_of_role|>system<|end_of_role|>You are helpful.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Hi<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
        $ids = self::$tokenizer->encode($prompt);
        $decoded = self::$tokenizer->decode($ids, skipSpecialTokens: false);
        $this->assertSame($prompt, $decoded);
    }
}
