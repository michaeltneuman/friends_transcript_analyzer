import pandas as pd
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import time
import io
import os
from urllib.parse import urlparse
from bs4 import BeautifulSoup

class FriendsDialogueAnalyzer:
    
    DATA_SOURCES = {
    'ederson_html': 'https://edersoncorbari.github.io/friends-scripts/season/{season:02d}{episode:02d}.html'
    }
    
    def __init__(self):
        self.main_characters = {
            'Rachel', 'Monica', 'Phoebe', 'Ross', 'Chandler', 'Joey'
        }
        self.character_aliases = {
            'Rachel': ['Rachel', 'Rach'],
            'Monica': ['Monica', 'Mon'],
            'Phoebe': ['Phoebe', 'Pheebs'],
            'Ross': ['Ross'],
            'Chandler': ['Chandler', 'Chan'],
            'Joey': ['Joey', 'Joe'],
            'All': ['All',]
        }
        self.alias_to_character = {}
        for char, aliases in self.character_aliases.items():
            for alias in aliases:
                self.alias_to_character[alias.upper()] = char
    
    def download_friends_data(self, season=1, episode=1, save_path=None):
        """Download and parse Friends script from HTML source"""
        if save_path is None:
            save_path = f'friends_s{season:02d}e{episode:02d}.csv'
        url = self.DATA_SOURCES['ederson_html'].format(season=season, episode=episode)
        while True:
            print(f"Downloading from: {url}")
            try:
                soup = BeautifulSoup(requests.get(url).content, 'html.parser')
                dialogue_data = []
                lines = str(soup).splitlines()
                current_speaker = None
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if ':</b>' in line:
                        line = line.split(':</b> ')
                        speakers = line[0].split('<b>')[-1].split(' and ')
                        dialogue = ':'.join(line[1:])
                        if dialogue:
                            for current_speaker in speakers:
                                dialogue_data.append({
                                    'season': season,
                                    'episode': episode,
                                    'speaker': current_speaker,
                                    'text': dialogue
                                })
                df = pd.DataFrame(dialogue_data)
                df.to_csv(save_path, index=False)
                print(f"✅ Downloaded {len(df)} lines to: {save_path}")
                if len(df) != 0:
                    return save_path
                print(f"❌ Didn't get enough, sleeping a bit then retrying...")
                time.sleep(3)
            except Exception as e:
                print(f"❌ Error: {e}")
                return None
    
    def load_or_download_data(self, local_path='friends_data.csv', force_download=False):
        """
        Load data from local file, or download if it doesn't exist
        
        Args:
            local_path (str): Path to local data file
            force_download (bool): Force re-download even if file exists
        
        Returns:
            pandas.DataFrame or None
        """
        if not force_download and os.path.exists(local_path):
            print(f"Loading existing data from: {local_path}")
            try:
                return pd.read_csv(local_path)
            except Exception as e:
                print(f"Error loading local file: {e}")
                print("Attempting to download fresh data...")
        
        downloaded_path = self.download_friends_data(save_path=local_path)
        if downloaded_path:
            try:
                return pd.read_csv(downloaded_path)
            except Exception as e:
                print(f"Error loading downloaded file: {e}")
                return None
        
        return None
        """Convert speaker name to standardized character name"""
        if not speaker:
            return "Unknown"
        
        clean_speaker = re.sub(r'[^\w\s]', '', speaker).strip().upper()
        
        if clean_speaker in self.alias_to_character:
            return self.alias_to_character[clean_speaker]
        
        return "Other"
    
    def count_words(self, text):
        """Count words in a text string"""
        return 0 if not text else len(re.sub(r'\([^)]*\)|\[[^\]]*\]', '', text).strip().split())
    
    def analyze_episode(self, season=1, episode=1, data_source=None):
        """Analyze a specific episode"""
        if data_source is None:
            data_source = self.download_friends_data(season=season, episode=episode)
            if data_source is None:
                return None
        df = pd.read_csv(data_source)
        if df is None:
            print("❌ Could not load transcript data")
            return None
        
        episode_df = self.filter_episode(df, season, episode)
        if episode_df.empty:
            print(f"❌ No data found for Season {season}, Episode {episode}")
            return None
        print(f"📺 Season {season}, Episode {episode} || Found {len(episode_df)} lines of dialogue")
        results = self._analyze_dataframe(episode_df)
        if results:
            results['episode_info'] = {
                'season': season,
                'episode': episode,
                'total_lines': len(episode_df)
            }
        return results
    
    def filter_episode(self, df, season, episode):
        """Filter dataframe to specific episode based on available columns"""
        season_cols = ['season', 'Season', 'season_num', 'season_number']
        episode_cols = ['episode', 'Episode', 'episode_num', 'episode_number']
        season_col = None
        episode_col = None
        
        for col in season_cols:
            if col in df.columns:
                season_col = col
                break
        
        for col in episode_cols:
            if col in df.columns:
                episode_col = col
                break
        
        if season_col is None or episode_col is None:
            print(f"Warning: Could not find season/episode columns. Available columns: {list(df.columns)}")
            print("Analyzing entire dataset...")
            return df
        
        filtered = df[(df[season_col] == season) & (df[episode_col] == episode)]
        return filtered
    def analyze_episode_from_csv(self, csv_file_path, season=None, episode=None):
        """
        Analyze dialogue from a CSV file containing Friends transcripts
        """
        try:
            df = pd.read_csv(csv_file_path)
            
            if season is not None and episode is not None:
                df = self.filter_episode(df, season, episode)
                if df.empty:
                    print(f"No data found for Season {season}, Episode {episode}")
                    return None
            
            return self._analyze_dataframe(df)
            
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None
    
    def normalize_speaker(self, speaker):
        """Convert speaker name to standardized character name"""
        if not speaker:
            return "Unknown"
        clean_speaker = re.sub(r'[^\w\s]', '', speaker).strip().upper()
        if clean_speaker in self.alias_to_character:
            return self.alias_to_character[clean_speaker]
        return "Other"
    
    def analyze_episode_from_text(self, text_content):
        """
        Analyze dialogue from raw text format
        Assumes format like:
        Speaker Name: Dialogue text
        """
        lines = text_content.strip().split('\n')
        dialogue_data = []
        
        for line in lines:
            match = re.match(r'^([^:]+):\s*(.+)$', line.strip())
            if match:
                speaker = match.group(1).strip()
                text = match.group(2).strip()
                dialogue_data.append({
                    'speaker': speaker,
                    'text': text
                })
        
        df = pd.DataFrame(dialogue_data)
        return self._analyze_dataframe(df)
    
    def create_visualization(self, results, episode_info="", save_path=None):
        """Create visualization showing both word counts and line counts"""
        if not results:
            return
        main_data = results['main_characters']
        word_counts = main_data['word_counts']
        line_counts = main_data['line_counts']
        other_words = results['other_characters']['total_words']
        other_lines = results['other_characters']['total_lines']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        chars = list(word_counts.keys())
        words = list(word_counts.values())
        
        bars1 = ax1.bar(chars, words, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_title(f'Word Count by Main Character {episode_info}')
        ax1.set_ylabel('Word Count')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        lines = list(line_counts.values())
        
        bars2 = ax2.bar(chars, lines, color='lightcoral', edgecolor='darkred', alpha=0.7)
        ax2.set_title(f'Line Count by Main Character {episode_info}')
        ax2.set_ylabel('Line Count')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        all_word_data = list(word_counts.values()) + ([other_words] if other_words > 0 else [])
        all_labels = list(word_counts.keys()) + (['Other'] if other_words > 0 else [])
        
        ax3.pie(all_word_data, labels=all_labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title(f'Word Distribution {episode_info}')
        
        all_line_data = list(line_counts.values()) + ([other_lines] if other_lines > 0 else [])
        
        ax4.pie(all_line_data, labels=all_labels, autopct='%1.1f%%', startangle=90)
        ax4.set_title(f'Line Distribution {episode_info}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    
    def _analyze_dataframe(self, df):
        """Core analysis logic for a DataFrame with speaker and text columns"""
        if df.empty:
            return None
        speaker_cols = ['speaker', 'Speaker', 'character', 'Character', 'who', 'name']
        text_cols = ['text', 'Text', 'dialogue', 'Dialogue', 'line', 'utterance']
        
        speaker_col = None
        text_col = None
        
        for col in speaker_cols:
            if col in df.columns:
                speaker_col = col
                break
        
        for col in text_cols:
            if col in df.columns:
                text_col = col
                break
        
        if speaker_col is None or text_col is None:
            print(f"Error: Could not identify speaker/text columns. Available columns: {list(df.columns)}")
            return None
        
        word_counts = defaultdict(int)
        line_counts = defaultdict(int)
        
        for _, row in df.iterrows():
            try:
                speaker = self.normalize_speaker(row.get(speaker_col, ''))
            except:
                continue
            text = row.get(text_col, '')
            
            if speaker.upper().strip() == 'ALL':
                words_per_char = word_count // len(self.main_characters)
                for main_char in self.main_characters:
                    word_counts[main_char] += words_per_char
                    line_counts[main_char] += 1
            else:
                word_count = self.count_words(text)
                word_counts[speaker] += word_count
                line_counts[speaker] += 1
        
        main_char_words = {char: word_counts[char] for char in self.main_characters}
        main_char_lines = {char: line_counts[char] for char in self.main_characters}
        
        other_words = sum(count for speaker, count in word_counts.items() 
                         if speaker not in self.main_characters and speaker != "Unknown")
        other_lines = sum(count for speaker, count in line_counts.items() 
                         if speaker not in self.main_characters and speaker != "Unknown")
        
        results = {
            'main_characters': {
                'word_counts': main_char_words,
                'line_counts': main_char_lines,
                'total_words': sum(main_char_words.values()),
                'total_lines': sum(main_char_lines.values())
            },
            'other_characters': {
                'total_words': other_words,
                'total_lines': other_lines
            },
            'overall_totals': {
                'total_words': sum(word_counts.values()),
                'total_lines': sum(line_counts.values())
            }
        }
        
        return results
    
    def print_analysis(self, results, episode_info=""):
        """Print formatted analysis results"""
        if not results:
            print("No results to display")
            return
        print(f"{'='*50}\nFRIENDS DIALOGUE ANALYSIS {episode_info}\n{'='*50}\n📊 MAIN CHARACTERS:")
        main_data = results['main_characters']
        sorted_chars = sorted(main_data['word_counts'].items(), 
                            key=lambda x: x[1], reverse=True)
        for char, words in sorted_chars:
            lines = main_data['line_counts'][char]
            avg_words = words / lines if lines > 0 else 0
            print(f"  {char:8}: {words:4} words, {lines:3} lines (avg: {avg_words:.1f} words/line)")
        print(f"📈 Main characters total: {main_data['total_words']} words, {main_data['total_lines']} lines")
        other_data = results['other_characters']
        if other_data['total_words'] > 0:
            avg_other = other_data['total_words'] / other_data['total_lines'] if other_data['total_lines'] > 0 else 0
            print(f"  👥 Other characters: {other_data['total_words']} words, {other_data['total_lines']} lines (avg: {avg_other:.1f} words/line)")
        totals = results['overall_totals']
        print(f"🎯 EPISODE TOTAL: {totals['total_words']} words, {totals['total_lines']} lines")
        print(f"\n📈 WORD DISTRIBUTION:")
        for char, words in sorted_chars:
            percentage = (words / totals['total_words']) * 100 if totals['total_words'] > 0 else 0
            print(f"  {char:8}: {percentage:.1f}%")

def quick_start_single_episode(season=1, episode=1):
    """Quick start function to analyze a single episode"""
    analyzer = FriendsDialogueAnalyzer()
    print(f"🎬 Starting analysis of Friends Season {season}, Episode {episode}")
    results = analyzer.analyze_episode(season=season, episode=episode)
    if results:
        episode_info = f"(Season {season}, Episode {episode})"
        analyzer.print_analysis(results, episode_info)
        analyzer.create_visualization(results, episode_info)
        return results
    else:
        print("Analysis failed. Please check the error messages above.")
        return None

def analyze_multiple_episodes(episodes_list):
    """
    Analyze multiple episodes and compare results
    
    Args:
        episodes_list: List of tuples [(season1, episode1), (season2, episode2), ...]
    """
    analyzer = FriendsDialogueAnalyzer()
    all_results = []
    
    for season, episode in episodes_list:
        print(f"\n🎬 Analyzing Season {season}, Episode {episode}")
        results = analyzer.analyze_episode(season=season, episode=episode)
        if results:
            results['season'] = season
            results['episode'] = episode
            all_results.append(results)
            episode_info = f"(S{season}E{episode})"
            analyzer.print_analysis(results, episode_info)
        else:
            print(f"❌ Failed to analyze Season {season}, Episode {episode}")
    
    if len(all_results) > 1:
        print(f"\n📊 COMPARISON SUMMARY")
        for result in all_results:
            s, e = result['season'], result['episode']
            main_words = result['main_characters']['total_words']
            print(f"S{s}E{e}: {main_words} total words by main characters")

def download_data_only():
    """Just download the Friends transcript data without analysis"""
    analyzer = FriendsDialogueAnalyzer()    
    print("📥 Downloading Friends transcript data...")
    data_path = analyzer.download_friends_data()    
    if data_path:
        print(f"\n✅ Data ready for analysis!\nFile location: {data_path}\n\nNext steps:\n1. Run: quick_start_single_episode(1, 1)\n2. Or: analyzer.analyze_episode(season=1, episode=1)")    
    return data_path

if __name__ == "__main__":
    print("🎭 FRIENDS DIALOGUE ANALYZER\n"+"=" * 60+"\nQUICK START OPTIONS:\n1. Analyze single episode:\n\tquick_start_single_episode(season=1, episode=1)")
    print("2. Analyze multiple episodes:\n\tanalyze_multiple_episodes([(1,1), (1,2), (2,1)])\n3. Just download data:\n\tdownload_data_only()\n4. Manual analysis:")
    print("\tanalyzer = FriendsDialogueAnalyzer()\n\tresults = analyzer.analyze_episode(season=1, episode=1)\n\tanalyzer.print_analysis(results)")
    for season,episode_max in {1:24,2:24,3:25,4:24,5:24,6:25,7:24,8:24,9:24,10:18}.items():
        for episode in range(1,episode_max+1):
            if (season,episode) in (
            (1,16),(1,17), # 2
            (2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(2,10),(2,11),(2,12),(2,13),(2,14),(2,15),(2,16),(2,17),(2,18),(2,19),(2,20),(2,21),(2,22),(2,23),(2,24), # 22
            (4,3),(4,6),(4,24), # 3
            (5,24), # 1
            (6,8),(6,15),(6,16),(6,25), # 4
            (7,24), # 1
            (8,13),(8,15),(8,24), # 3
            (9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,9),(9,13),(9,15),(9,16),(9,18),(9,23),(9,24), # 13
            (10,1),(10,2),(10,3),(10,4),(10,5),(10,6),(10,7),(10,8),(10,9),(10,10),(10,11),(10,12),(10,13),(10,14),(10,15),(10,16),(10,17),(10,18) # 18
            ):
                # 67 episodes left to learn how to parse
                continue
            quick_start_single_episode(season,episode)
            exit()