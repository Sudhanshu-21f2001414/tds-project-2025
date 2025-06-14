import os
import json
import markdown
from bs4 import BeautifulSoup

COURSE_DIR = "data/course_content"
DISCOURSE_DIR = "data/discourse"
OUTPUT_FILE = "data/scraped_content.json"

merged_content = []

def extract_text_from_markdown(md_content):
    html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")

def load_course_markdowns():
    entries = []
    for root, _, files in os.walk(COURSE_DIR):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    text = extract_text_from_markdown(f.read())
                    entries.append({
                        "source": "course",
                        "text": text,
                        "url": f"file://{path}"
                    })
    print(f"✅ Loaded {len(entries)} course content entries.")
    return entries

def load_discourse_json():
    entries = []
    for root, _, files in os.walk(DISCOURSE_DIR):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            posts = data.get("post_stream", {}).get("posts", [])
                        elif isinstance(data, list):
                            posts = data
                        else:
                            posts = []

                        for post in posts:
                            cleaned = BeautifulSoup(post.get("cooked", ""), "html.parser").get_text(separator="\n")
                            entries.append({
                                "source": "discourse",
                                "text": cleaned,
                                "url": f"https://discourse.onlinedegree.iitm.ac.in/t/{post.get('topic_slug', 'unknown')}/{post.get('topic_id', 0)}/{post.get('post_number', 1)}"
                            })
                except Exception as e:
                    print(f"⚠️ Error reading {file}: {e}")
    print(f"✅ Loaded {len(entries)} discourse entries.")
    return entries

def main():
    print("🔄 Loading content...")
    course_entries = load_course_markdowns()
    discourse_entries = load_discourse_json()

    merged = course_entries + discourse_entries
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"✅ Merged {len(merged)} entries into {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
