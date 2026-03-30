<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Generation;

/**
 * ByteLevel BPE tokenizer compatible with HuggingFace tokenizer.json format.
 *
 * Implements the same tokenization algorithm as GPT-2/Granite models:
 * 1. Pre-tokenize using byte-level regex splitting
 * 2. Convert bytes to unicode characters
 * 3. Apply BPE merges iteratively
 * 4. Map tokens to IDs via vocabulary
 */
final class BpeTokenizer
{
    /** @var array<string, int> Token string to ID mapping */
    private array $vocab;

    /** @var array<int, string> ID to token string mapping */
    private array $idToToken;

    /** @var array<string, int> Merge pair to priority mapping (lower = higher priority) */
    private array $mergeRanks;

    /** @var array<string, int> Special token string to ID mapping */
    private array $specialTokens;

    /** @var array<int, string> Byte value to unicode character mapping */
    private array $byteEncoder;

    /** @var array<string, int> Unicode character to byte value mapping */
    private array $byteDecoder;

    /** @var string Regex pattern for pre-tokenization (GPT-2 style) */
    private string $pattern;

    private function __construct(
        array $vocab,
        array $mergeRanks,
        array $specialTokens,
    ) {
        $this->vocab = $vocab;
        $this->idToToken = array_flip($vocab);
        $this->mergeRanks = $mergeRanks;
        $this->specialTokens = $specialTokens;

        // Add special tokens to the reverse mapping
        foreach ($specialTokens as $content => $id) {
            $this->idToToken[$id] = $content;
        }

        [$this->byteEncoder, $this->byteDecoder] = self::buildByteMapping();

        // GPT-2 pre-tokenization regex (with unicode support)
        $this->pattern = "/('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)/u";
    }

    /**
     * Load tokenizer from a HuggingFace tokenizer.json file.
     */
    public static function fromFile(string $path): self
    {
        $json = json_decode(file_get_contents($path), true, 512, JSON_THROW_ON_ERROR);

        if (($json['model']['type'] ?? '') !== 'BPE') {
            throw new \InvalidArgumentException("Only BPE tokenizers are supported, got: " . ($json['model']['type'] ?? 'unknown'));
        }

        $vocab = $json['model']['vocab'];

        $mergeRanks = [];
        foreach ($json['model']['merges'] as $i => $merge) {
            // Merges can be either [a, b] arrays or "a b" strings
            if (is_array($merge)) {
                $key = $merge[0] . ' ' . $merge[1];
            } else {
                $key = $merge;
            }
            $mergeRanks[$key] = $i;
        }

        $specialTokens = [];
        foreach ($json['added_tokens'] ?? [] as $token) {
            if ($token['special'] ?? false) {
                $specialTokens[$token['content']] = $token['id'];
            }
        }

        return new self($vocab, $mergeRanks, $specialTokens);
    }

    /**
     * Encode text to token IDs.
     *
     * @return int[]
     */
    public function encode(string $text): array
    {
        if ($text === '') {
            return [];
        }

        $ids = [];

        // Handle special tokens by splitting around them
        $segments = $this->splitOnSpecialTokens($text);

        foreach ($segments as [$segment, $isSpecial]) {
            if ($isSpecial) {
                $ids[] = $this->specialTokens[$segment];
                continue;
            }

            if ($segment === '') {
                continue;
            }

            // Pre-tokenize: split into words using GPT-2 regex
            preg_match_all($this->pattern, $segment, $matches);
            $words = $matches[0];

            foreach ($words as $word) {
                // Convert each byte to its unicode representation
                $unicodeWord = '';
                $bytes = unpack('C*', $word);
                foreach ($bytes as $byte) {
                    $unicodeWord .= $this->byteEncoder[$byte];
                }

                // Apply BPE
                $bpeTokens = $this->bpe($unicodeWord);

                foreach ($bpeTokens as $token) {
                    if (isset($this->vocab[$token])) {
                        $ids[] = $this->vocab[$token];
                    }
                    // Unknown tokens are silently dropped (rare with byte-level BPE)
                }
            }
        }

        return $ids;
    }

    /**
     * Decode token IDs back to text.
     *
     * @param int[] $ids
     */
    public function decode(array $ids, bool $skipSpecialTokens = true): string
    {
        $result = '';

        foreach ($ids as $id) {
            if (!isset($this->idToToken[$id])) {
                continue;
            }
            $token = $this->idToToken[$id];
            $isSpecial = isset($this->specialTokens[$token]);

            if ($skipSpecialTokens && $isSpecial) {
                continue;
            }

            if ($isSpecial) {
                // Special tokens are emitted as-is (not byte-decoded)
                $result .= $token;
            } else {
                // Regular tokens: decode byte-level characters back to actual bytes
                $bytes = [];
                $token = (string) $token;
                $chars = mb_str_split($token);
                foreach ($chars as $char) {
                    if (isset($this->byteDecoder[$char])) {
                        $bytes[] = $this->byteDecoder[$char];
                    }
                }
                if (!empty($bytes)) {
                    $result .= pack('C*', ...$bytes);
                }
            }
        }

        return $result;
    }

    /**
     * Get the vocabulary size.
     */
    public function vocabSize(): int
    {
        return count($this->vocab) + count($this->specialTokens);
    }

    /**
     * Get a special token ID.
     */
    public function specialTokenId(string $token): ?int
    {
        return $this->specialTokens[$token] ?? $this->vocab[$token] ?? null;
    }

    /**
     * Apply BPE merges to a unicode-encoded word.
     *
     * @return string[] List of BPE tokens
     */
    private function bpe(string $word): array
    {
        $chars = mb_str_split($word);
        if (count($chars) <= 1) {
            return $chars;
        }

        // Start with individual characters as the initial tokens
        $pieces = $chars;

        while (count($pieces) > 1) {
            // Find the highest priority merge pair
            $bestRank = PHP_INT_MAX;
            $bestIdx = -1;

            for ($i = 0; $i < count($pieces) - 1; $i++) {
                $pair = $pieces[$i] . ' ' . $pieces[$i + 1];
                $rank = $this->mergeRanks[$pair] ?? PHP_INT_MAX;
                if ($rank < $bestRank) {
                    $bestRank = $rank;
                    $bestIdx = $i;
                }
            }

            if ($bestIdx === -1) {
                break; // No more merges possible
            }

            // Apply the merge
            $merged = $pieces[$bestIdx] . $pieces[$bestIdx + 1];
            array_splice($pieces, $bestIdx, 2, [$merged]);
        }

        return $pieces;
    }

    /**
     * Split text on special tokens, preserving them as separate segments.
     *
     * @return array<array{0: string, 1: bool}> [segment, isSpecial] pairs
     */
    private function splitOnSpecialTokens(string $text): array
    {
        if (empty($this->specialTokens)) {
            return [[$text, false]];
        }

        // Sort special tokens by length (longest first) to avoid partial matches
        $tokens = array_keys($this->specialTokens);
        usort($tokens, fn(string $a, string $b) => mb_strlen($b) - mb_strlen($a));

        $pattern = '/(' . implode('|', array_map('preg_quote', $tokens)) . ')/u';

        $parts = preg_split($pattern, $text, -1, PREG_SPLIT_DELIM_CAPTURE);
        $segments = [];

        foreach ($parts as $part) {
            if ($part === '') {
                continue;
            }
            $segments[] = [$part, isset($this->specialTokens[$part])];
        }

        return $segments;
    }

    /**
     * Build the byte-to-unicode mapping used by ByteLevel BPE.
     *
     * This is the same mapping as in GPT-2's bytes_to_unicode() function.
     * Maps bytes 0-255 to unicode characters, using printable ASCII directly
     * and offsetting control characters to avoid issues.
     *
     * @return array{0: array<int, string>, 1: array<string, int>}
     */
    private static function buildByteMapping(): array
    {
        $byteEncoder = [];
        $n = 0;

        // Printable ASCII-ish ranges that map to themselves
        // ! to ~ (33-126)
        for ($b = ord('!'); $b <= ord('~'); $b++) {
            $byteEncoder[$b] = mb_chr($b);
        }
        // ¡ to ¬ (161-172)
        for ($b = 0xA1; $b <= 0xAC; $b++) {
            $byteEncoder[$b] = mb_chr($b);
        }
        // ® to ÿ (174-255)
        for ($b = 0xAE; $b <= 0xFF; $b++) {
            $byteEncoder[$b] = mb_chr($b);
        }

        // Everything else gets mapped to 256+
        $offset = 256;
        for ($b = 0; $b < 256; $b++) {
            if (!isset($byteEncoder[$b])) {
                $byteEncoder[$b] = mb_chr($offset);
                $offset++;
            }
        }

        $byteDecoder = [];
        foreach ($byteEncoder as $byte => $char) {
            $byteDecoder[$char] = $byte;
        }

        return [$byteEncoder, $byteDecoder];
    }
}
