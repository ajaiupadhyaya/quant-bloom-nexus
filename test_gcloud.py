# Import the correct exception libraries
import google.auth
from google.cloud import storage
from google.api_core import exceptions

try:
    # The client library will automatically find your credentials, which should
    # now include the quota project you just set.
    storage_client = storage.Client()

    print("✅ Successfully authenticated!")
    print("🔎 Listing buckets in your project:")

    buckets = storage_client.list_buckets()
    bucket_list = [bucket.name for bucket in buckets]

    if not bucket_list:
        print("--> You don't have any Cloud Storage buckets in this project yet.")
        print(
            "--> You can create one with the command: gsutil mb gs://your-unique-bucket-name"
        )
    else:
        for bucket_name in bucket_list:
            print(f"--> gs://{bucket_name}")

# This is the corrected way to catch a credentials error
except google.auth.exceptions.DefaultCredentialsError:
    print(
        "❌ Authentication failed. Please run 'gcloud auth application-default login' in your terminal."
    )

# This catches the "API Not Enabled" or "Project Not Found" type of error
except exceptions.Forbidden as e:
    print(f"❌ A permissions error occurred: {e}")
    print("\n💡 TIP: Make sure the 'Cloud Storage API' is enabled for your project.")
    print(
        "You can enable it here: https://console.cloud.google.com/apis/library/storage.googleapis.com"
    )

# A general catch-all for other errors
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
