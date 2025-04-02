import argparse
from src.bfxapi.lob_recorder import LobRecorder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a path string to find occurrences of a symbol.")
    parser.add_argument("--symbol", type=str, default="tBTCUSD", help="The symbol to search collect data. Default is 'tBTCUSD'.")
    parser.add_argument("--path", type=str, default="lob_data", help="The path to save data. Default is 'lob_data'.")
    
    args = parser.parse_args()

    lob_recorder = LobRecorder(args.symbol, saving_path = args.path)
    lob_recorder.run()
