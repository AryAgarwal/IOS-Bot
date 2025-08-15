

# iOS 26 Q&A CLI

A simple command-line tool to answer questions about iOS 26 using document indexing, retrieval, and optional web scraping.

## Features
- **Index Building**: Creates a TF-IDF index from provided iOS 26 documents.
- **Question Answering**: Retrieves the most relevant document snippets to answer your query.
- **Source Linking**: Displays source URLs and metadata for transparency.
- **Web Scraping**: Fetches latest iOS 26-related articles automatically before indexing.

## Tech Stack
- **Python** — Core programming language.
- **scikit-learn** — TF-IDF vectorization and cosine similarity for retrieval.
- **BeautifulSoup4 + Requests** — Web scraping articles about iOS 26.
- **JSON** — Storing the index and metadata.
- **CLI Interface** — Using `argparse` for command-line usage.

## Installation
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd ios26-qa
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Build the index
```bash
python ios26-chat.py --build_index
```
This reads the source documents (and optionally scrapes fresh articles) and saves a TF-IDF index to disk.

### Step 2: Ask a question
```bash
python ios26-chat.py --ask "What are the major features of iOS 26?"
```
The script will:
- Retrieve relevant text snippets.
- Generate an answer.
- List top sources.

## Example Output
```
=== Answer ===

**Official (Apple) [3, 4, 5]:**

*   A major update bringing a "beautiful new design," intelligent experiences, and improvements to commonly used apps.
*   Features a new "Liquid Glass" design, making apps and system experiences more expressive and delightful.
*   Includes significant updates to Phone and Messages apps to improve connectivity and reduce distractions.
*   Introduces new features in CarPlay, Apple Music, Maps, Wallet, and a brand-new Apple Games app.
*   Boasts enhanced Apple Intelligence capabilities integrated across the system.


**Third-party reporting [1, 2, 6, 7, 8]:**  (Note: these are beta features and subject to change)

*   Improved app animations with faster opening and closing speeds.
*   Changes to the Preview app with larger buttons and repositioned scanning buttons.
*   Removal of the Classic Mode toggle in the Camera app; Classic Mode is now the default.
*   New introductory video upon updating to iOS 26.
*   Several new ringtones added, including variants of the Reflection ringtone and a new Little Bird ringtone.
*  Updated Liquid Glass effects on Lock Screen, toggles, and navigation bars.

=== Top Sources ===
[1] Everything New in iOS 26 Beta 6 - MacRumors  (2025/08/11)  https://www.macrumors.com/2025/08/11/ios-26-beta-6-features/
[2] iOS 26 and iPadOS 26 public beta 3 now available - 9to5Mac  (2025/08/14)  https://9to5mac.com/2025/08/14/ios-26-ipados-26-public-beta-3/
[3] Apple elevates the iPhone experience with iOS 26 - Apple  (2025-08-15)  https://www.apple.com/newsroom/2025/06/apple-elevates-the-iphone-experience-with-ios-26/
[4] Apple elevates the iPhone experience with iOS 26 - Apple  (2025-08-15)  https://www.apple.com/newsroom/2025/06/apple-elevates-the-iphone-experience-with-ios-26/
[5] Apple elevates the iPhone experience with iOS 26 - Apple  (2025-08-15)  https://www.apple.com/newsroom/2025/06/apple-elevates-the-iphone-experience-with-ios-26/
[6] Apple Releases Third iOS 26 and iPadOS 26 Public Betas, New Developer Beta - MacRumors  (2025/08/14)  https://www.macrumors.com/2025/08/14/apple-releases-ios-26-public-beta-3/
[7] iOS 26 guide: All the new features for your iPhone and how to use them | Tom's Guide  (2025-07-26)  https://www.tomsguide.com/phones/iphones/ios-26-guide
[8] Everything New in iOS 26 Beta 6 - MacRumors  (2025/08/11)  https://www.macrumors.com/2025/08/11/ios-26-beta-6-features/

```

## Converting to a Streamlit App
This can also be extended to have a streamlit frontend for better UX.


