import os
import sys
import random
import time
from threading import Thread

# Ensure we can import infer.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from app.models.infer import predict

def simulate_realtime_pipeline():
    print("==================================================")
    print("🚀 REAL-TIME SITUATION CLASSIFIER SIMULATOR")
    print("==================================================")
    print("Type an emergency transcript below (or type 'exit' to quit).")
    print("The pipeline will instantly process it through the loaded AI model.\n")

    # Synthetic test pool for evaluating robustness
    emotions = ["fear", "anger", "panic", "neutral", "sadness", "surprise"]
    sounds = ["scream", "siren", "glass_shatter", "gunshot", "crying", "explosion", "none"]

    while True:
        try:
            # Capture real-time input (simulating live speech-to-text)
            transcript = input("🎤 [Live Audio Transcript] > ")
            if transcript.lower() in ['exit', 'quit']:
                print("Stopping real-time feed...")
                break
            
            if not transcript.strip():
                continue

            # Synthesize upstream perception layers for local testing
            simulated_emotion = random.choice(emotions)
            simulated_sounds = random.sample(sounds, k=random.randint(0, 2))

            perception_output = {
                "transcript": transcript,
                "sound_events": simulated_sounds,
                "emotion": simulated_emotion,
                "speakers": 1
            }

            print("\n⚙️  Processing Perception Output:")
            print(f"   - Transcript: '{transcript}'")
            print(f"   - Emotion:    {simulated_emotion}")
            print(f"   - Sounds:     {simulated_sounds}")

            # Measure real-time inference latency
            start_time = time.time()
            
            prediction = predict(perception_output)
            
            latency = (time.time() - start_time) * 1000 # in ms

            print(f"\n🚨 >> PREDICTION: {prediction['label'].upper()} <<")
            print(f"📊 Confidence: {prediction['confidence']:.2f}")
            print(f"⏱️  Latency:    {latency:.2f} ms\n")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nStopping real-time feed...")
            break

if __name__ == "__main__":
    simulate_realtime_pipeline()
