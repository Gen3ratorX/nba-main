#!/usr/bin/env python3
"""
setup_odds_api.py - Quick Setup for The Odds API
Run this to configure your API key and test the connection
"""
import os
import sys
from pathlib import Path

def setup_api_key():
    """Interactive setup for API key"""
    print("\n" + "="*70)
    print("🔧 THE ODDS API - SETUP")
    print("="*70 + "\n")
    
    # Check if .env exists
    env_file = Path('.env')
    
    # Check for existing key
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if 'ODDS_API_KEY=' in content and 'your_key_here' not in content:
                print("✅ API key already configured in .env file")
                
                # Extract and show (masked)
                for line in content.split('\n'):
                    if line.startswith('ODDS_API_KEY='):
                        key = line.split('=')[1].strip()
                        masked = key[:8] + '*' * (len(key) - 12) + key[-4:]
                        print(f"   Current key: {masked}")
                
                response = input("\n   Update API key? (y/n): ").lower()
                if response != 'y':
                    return True
    
    print("""
📚 GETTING YOUR FREE API KEY:

1. Go to: https://the-odds-api.com/
2. Click "Get Started Free"
3. Sign up with email
4. Copy your API key from the dashboard

Free Tier Includes:
• 500 requests per month
• Updates every 5 minutes
• All major US sportsbooks
• Moneyline, spreads, and totals

""")
    
    api_key = input("Paste your API key here: ").strip()
    
    if not api_key:
        print("❌ No API key entered")
        return False
    
    if len(api_key) < 20:
        print("⚠️  That doesn't look like a valid API key (too short)")
        response = input("   Continue anyway? (y/n): ").lower()
        if response != 'y':
            return False
    
    # Save to .env
    if env_file.exists():
        # Read existing content
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Remove old ODDS_API_KEY line if exists
        lines = [l for l in lines if not l.startswith('ODDS_API_KEY=')]
        
        # Add new key
        lines.append(f"\nODDS_API_KEY={api_key}\n")
        
        with open(env_file, 'w') as f:
            f.writelines(lines)
    else:
        # Create new .env file
        with open(env_file, 'w') as f:
            f.write(f"# The Odds API Configuration\n")
            f.write(f"ODDS_API_KEY={api_key}\n")
    
    print(f"\n✅ API key saved to .env file")
    return True

def test_connection():
    """Test API connection"""
    print("\n" + "="*70)
    print("🧪 TESTING API CONNECTION")
    print("="*70 + "\n")
    
    try:
        from vegas_odds_fetcher import VegasOddsFetcher
        
        fetcher = VegasOddsFetcher()
        
        if not fetcher.api_key:
            print("❌ No API key found - setup may have failed")
            return False
        
        print("📡 Fetching NBA odds...")
        odds = fetcher.get_nba_odds()
        
        if not odds:
            print("❌ Failed to fetch odds")
            print("   Possible issues:")
            print("   • Invalid API key")
            print("   • No NBA games today")
            print("   • Rate limit exceeded")
            return False
        
        print(f"✅ Successfully connected!")
        print(f"   Games fetched: {len(odds.get('games', []))}")
        print(f"   Requests remaining: {odds.get('requests_remaining', 'unknown')}")
        
        # Display odds
        if odds.get('games'):
            fetcher.display_odds_summary()
        
        return True
    
    except ImportError:
        print("❌ vegas_odds_fetcher.py not found")
        print("   Make sure it's in the same directory")
        return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_integration():
    """Check if integration is ready"""
    print("\n" + "="*70)
    print("✅ INTEGRATION CHECKLIST")
    print("="*70 + "\n")
    
    checks = []
    
    # Check 1: .env file exists
    if Path('.env').exists():
        checks.append(("✅", ".env file exists"))
    else:
        checks.append(("❌", ".env file missing"))
    
    # Check 2: API key is set
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('ODDS_API_KEY')
    if api_key and api_key != 'your_key_here':
        checks.append(("✅", "API key configured"))
    else:
        checks.append(("❌", "API key not set"))
    
    # Check 3: vegas_odds_fetcher.py exists
    if Path('vegas_odds_fetcher.py').exists():
        checks.append(("✅", "vegas_odds_fetcher.py present"))
    else:
        checks.append(("❌", "vegas_odds_fetcher.py missing"))
    
    # Check 4: main_v2.py exists
    if Path('main_v2.py').exists():
        checks.append(("✅", "main_v2.py present"))
    else:
        checks.append(("⚠️", "main_v2.py not found (using main.py?)"))
    
    for status, message in checks:
        print(f"   {status} {message}")
    
    print("\n" + "="*70)
    
    all_passed = all(status == "✅" for status, _ in checks)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Ready to integrate!")
        print("\nNext steps:")
        print("1. Follow INTEGRATION_GUIDE.md to update main_v2.py")
        print("2. Run: python main.py --bankroll 1000")
        print("3. You should now see real Vegas lines!")
    else:
        print("⚠️  Some checks failed - review above")
    
    return all_passed

def main():
    """Main setup flow"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                  THE ODDS API - SETUP WIZARD                  ║
    ║                                                               ║
    ║  This will help you configure Vegas odds fetching for        ║
    ║  your NBA prediction system.                                 ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Setup API key
    if not setup_api_key():
        print("\n❌ Setup cancelled or failed")
        sys.exit(1)
    
    # Step 2: Test connection
    if not test_connection():
        print("\n⚠️  Connection test failed")
        print("   You can still use the system with default lines (228.0)")
        response = input("\n   Continue with integration anyway? (y/n): ").lower()
        if response != 'y':
            sys.exit(1)
    
    # Step 3: Verify integration readiness
    verify_integration()
    
    print("\n" + "="*70)
    print("✅ SETUP COMPLETE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)