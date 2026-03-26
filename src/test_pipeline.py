import os
from data.data_streamer import ParquetDataStreamer
from model.ids_cnn import IntrusionDetectionCNN

def run_integration_test():
    print("--- Starting V2 Pipeline Integration Test ---")
    
    # 1. Locate your data file
    # Make sure this path points to your actual parquet file!
    data_path = "../data/sampled_data.parquet" 
    
    if not os.path.exists(data_path):
        print(f"ERROR: Cannot find data at {data_path}. Please check the path.")
        return

    # 2. Initialize the Data Streamer (Streaming Batch Size = 32)
    print("Initializing Data Streamer...")
    streamer = ParquetDataStreamer(file_path=data_path, batch_size=32, num_clients=1, client_id=0)
    tf_dataset = streamer.get_tf_dataset()

    # 3. Initialize the 1D-CNN
    # We use shape (len(features), 1) to match the 1D-CNN input requirement
    print("Initializing 1D-CNN Model...")
    input_shape = (len(streamer.features), 1)
    cnn = IntrusionDetectionCNN(input_shape=input_shape, num_classes=2) # 2 for Binary classification test

    # 4. Train on the Stream!
    print("Attempting to train 1 epoch on the stream (Watch your RAM!)...")
    try:
        cnn.train_on_stream(tf_dataset, epochs=1)
        print("\nSUCCESS! The CNN successfully trained on the data stream without loading it all into memory.")
    except Exception as e:
        print(f"\nFAILED: An error occurred: {e}")

if __name__ == "__main__":
    run_integration_test()