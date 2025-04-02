import argparse
from src.bfxapi.lob_data_converter import LobDataConverter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a path string to find occurrences of a symbol.")
    parser.add_argument("--prefix_path", type=str, help="Path where lob files are stored.")
    parser.add_argument("--subdirectory", type=str, default="lob_data", help="Name of subdirectory to save formatted data")
    
    args = parser.parse_args()

    lob_converter = LobDataConverter(args.prefix_path, args.subdirectory)
    lob_converter.convert_files()