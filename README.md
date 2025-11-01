# Agency

Hey there! This is my collection of random agent architectures I've been playing around with. Think of it as my personal lab where I experiment with different ways AI agents can work.

## What's Inside

I've built two different web search agents so far, each with its own approach to finding and processing information from the web.

### Simple Web Search Agent

This one's pretty straightforward. I designed it to be a solid research assistant that knows how to dig deep and write comprehensive reports.

How it works: The agent has two main tools at its disposal - web search and page scraping. When you ask it something, it searches the web using multiple queries at once (up to 3), then actually scrapes the content from relevant pages to get the full story, not just snippets. I built it to always do multiple rounds of research because I wanted it to cross-reference sources and really explore topics from different angles.

The cool part is I programmed it to write like a human would - conversational but detailed. It uses headings to organize thoughts, throws in bullet points when they make sense, and most importantly, it writes in paragraphs. I made it target around 800-1500 words minimum because I hate those superficial summaries that don't actually tell you anything useful.

The agent follows this reflection process where before each action it basically thinks out loud about what it needs to do next. It'll say things like "I need to search for comprehensive information" or "I should scrape these sources to gather detailed data". Makes it feel more thoughtful somehow.

### Tool Reasoning Web Search Agent

This is the beefed up version. 

Main difference: I added explicit reasoning and done tools. The reasoning tool lets the agent actually articulate its research strategy before taking action. It's like watching someone think through their approach before diving in. The done tool signals when research is complete and it's ready to write the final report.

I made this one way more thorough in its research protocol. It has to do minimum 4-6 research rounds covering different dimensions - foundational understanding, technical deep dives, practical applications, comparative analysis, challenges and limitations, future trajectory. Each round involves searching and scraping multiple sources.

The writing requirements are intense here. I set the minimum at 1500 words but really I want 2500-3500 words for most topics. About 85% should be dense paragraphs (5-8 sentences each), only 10% lists and 5% tables. I basically told it "if you think you've written enough, write more".

The big philosophy here is dynamic adaptation. I drilled into the prompt that every research query is unique and it needs to completely adapt its approach. No generic templates. If you're researching quantum physics vs ancient history vs cooking techniques, the sections and structure should be totally different.

## Architecture Overview

Both agents use LangGraph under the hood with a pretty simple flow:
- Start node kicks things off
- LLM node processes requests and decides what tools to use
- Tools node executes the actual tool calls (searching, scraping, reasoning)
- Conditional edges determine if we keep using tools or end the graph
- Everything loops back to LLM until research is complete

They both use GPT-4.1 mini at 0.7 temperature, pulling from Serper API for search and scrape functionality.

## Why I Built These

I got tired of agents that give you surface level answers. These are designed to actually research topics properly - multiple rounds, diverse sources, cross-referencing, the works. The simple one is great for when you need solid comprehensive answers. The reasoning one is for when you really want to go deep and understand something thoroughly.

## Tech Stack

LangChain for the framework, LangGraph for agent orchestration, OpenAI for the LLM, Serper for web search and scraping. All written in TypeScript because I don't know.

## What's Next

I'll probably keep adding more agent patterns as I think of them. Maybe something with memory, or multi-agent collaboration, or specialized domain agents. This is just a playground for ideas really.

Feel free to poke around the code if you're curious about the implementations. Each agent has its own directory with the full code in index.ts.
