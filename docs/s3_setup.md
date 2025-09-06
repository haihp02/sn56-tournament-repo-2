### 1. Create a Bucket in Backblaze B2 (example)

#### Log in to Your Backblaze Account
- Navigate to the [Backblaze Sign-In Page](https://www.backblaze.com/sign-up/cloud-storage) and signup.
#### Access B2 Cloud Storage
- In the left-hand navigation pane, click on **"Buckets"** under the **"B2 Cloud Storage"** section.

#### Create a New Bucket
1. Click the **"Create a Bucket"** button.
2. Enter a unique **Bucket Name**.
3. Choose the **Bucket Type** (Private or Public).
4. Click **"Create Bucket"** to finalize.

### 2. Generate Application Keys

**Application keys in Backblaze B2 serve as the Access Key and Secret Key for S3-compatible interactions.**

#### Navigate to Application Keys
- In the left-hand navigation pane, click on **"Application Keys"**

#### Create a New Application Key
1. Click **"Add a New Application Key"**.
2. Provide a **Name** for the key.
3. Select the **Bucket** this key will have access to (or choose **"All"** for universal access).
4. Set the **Permissions** (Read and Write).
5. Optionally, specify a **File Name Prefix** to restrict access to files with that prefix.
6. Click **"Create New Key"**.

#### Save Your Keys
After creation, you'll receive:
- **KeyID**: This acts as the **Access Key**.
- **Application Key**: This serves as the **Secret Key**.

**Important**: The **Application Key** is displayed only once. Ensure you copy and store it securely.

#### 3. Determine Your S3-Compatible Endpoint

Backblaze B2 provides S3-compatible endpoints based on the region of your bucket.

### Locate the Endpoint URL
1. In the **"Buckets"** section, find your bucket and note its **"Endpoint"**.
2. The endpoint typically follows this format:
```plaintext
s3.<region>.backblazeb2.com
```

If your bucket is in the us-west-002 region, your endpoint would be:
```
s3.us-west-002.backblazeb2.com
```
