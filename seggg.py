import os

def get_image(input_image_name):
    """
    Checks if the input image exists in both the Google map folder and the accounts folder.
    Returns the path of the matched image from the accounts folder if found.
    
    Parameters:
        input_image_name (str): The name of the image to check for matching.

    Returns:
        str or None: The path to the matching image in the accounts folder, or None if no match is found.
    """
    google_map_folder = r"H:\fyp\data_1\New_dt_fyp\Google map"
    accounts_folder = r"H:\accounts"

    google_map_image_path = os.path.join(google_map_folder, input_image_name)

    if os.path.isfile(google_map_image_path):
        account_image_path = os.path.join(accounts_folder, input_image_name)

        if os.path.isfile(account_image_path):
            print(f"Matching account image found: {account_image_path}")
            return account_image_path
        else:
            print("No matching image found in accounts folder.")
            return None
    else:
        print("Image not found in Google map folder.")
        return None

def store_image_path(input_image_name):
    """
    Attempts to get the path of the matched image based on the input image name.
    Stores the path if a match is found.

    Parameters:
        input_image_name (str): The name of the image to match and store.

    Returns:
        str or None: The path of the matched image if found, otherwise None.
    """
    matched_path = get_image(input_image_name)

    if matched_path:
        saved_path = matched_path
        print(f"Saved path: {saved_path}")
        return saved_path
    else:
        print("No matching image found to store.")
        return None

def check_1(input_image_name):

    google_map_folder = r"H:\fyp\data_1\New_dt_fyp\Google map"
    accounts_folder = r"H:\accounts"

    google_map_image_path = os.path.join(google_map_folder, input_image_name)
    account_image_path = os.path.join(accounts_folder, input_image_name)

    return os.path.isfile(google_map_image_path) and os.path.isfile(account_image_path)
