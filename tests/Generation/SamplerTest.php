<?php

declare(strict_types=1);

namespace PhpMlKit\ONNXRuntime\Tests\Generation;

use PhpMlKit\ONNXRuntime\Generation\Sampler;
use PHPUnit\Framework\TestCase;

class SamplerTest extends TestCase
{
    public function testGreedySampling(): void
    {
        $sampler = Sampler::greedy();
        $logits = [0.1, 0.5, 0.9, 0.3, 0.2];
        $token = $sampler->sample($logits);
        $this->assertSame(2, $token); // Index of 0.9
    }

    public function testGreedyWithNegativeLogits(): void
    {
        $sampler = Sampler::greedy();
        $logits = [-1.0, -0.5, -2.0, -0.1];
        $token = $sampler->sample($logits);
        $this->assertSame(3, $token); // Index of -0.1 (largest)
    }

    public function testTemperatureSampling(): void
    {
        // With very low temperature, should behave like greedy
        $sampler = new Sampler(temperature: 0.001);
        $logits = [0.1, 0.5, 10.0, 0.3];
        $token = $sampler->sample($logits);
        $this->assertSame(2, $token);
    }

    public function testTopKSampling(): void
    {
        $sampler = new Sampler(temperature: 0.001, topK: 2);
        $logits = [5.0, 10.0, 3.0, 8.0];
        $token = $sampler->sample($logits);
        // Should pick from top-2: indices 1 (10.0) and 3 (8.0)
        $this->assertContains($token, [1, 3]);
    }

    public function testRepetitionPenalty(): void
    {
        $sampler = Sampler::greedy();
        $logits = [5.0, 4.9, 3.0];

        // Without penalty, picks index 0
        $this->assertSame(0, $sampler->sample($logits));

        // With penalty on index 0, should pick index 1
        $sampler = new Sampler(temperature: 0.0, repetitionPenalty: 2.0);
        $token = $sampler->sample($logits, generatedIds: [0]);
        $this->assertSame(1, $token);
    }

    public function testCreativePreset(): void
    {
        $sampler = Sampler::creative();
        // Should not throw, and should return a valid index
        $logits = array_fill(0, 100, 1.0);
        $token = $sampler->sample($logits);
        $this->assertGreaterThanOrEqual(0, $token);
        $this->assertLessThan(100, $token);
    }

    public function testLargeLogitsArray(): void
    {
        // Simulate vocab-sized logits (49155)
        $sampler = Sampler::greedy();
        $logits = array_fill(0, 49155, 0.0);
        $logits[12345] = 10.0; // Make one token much more likely
        $token = $sampler->sample($logits);
        $this->assertSame(12345, $token);
    }
}
