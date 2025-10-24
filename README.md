# M.Tech AI & ML Knowledge Base 🎓

[![MkDocs](https://img.shields.io/badge/MkDocs-526CFE?style=for-the-badge&logo=MaterialForMkDocs&logoColor=white)](https://www.mkdocs.org/)
[![Material Theme](https://img.shields.io/badge/Material-for%20MkDocs-blue?style=for-the-badge)](https://squidfunk.github.io/mkdocs-material/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

A comprehensive, semester-wise knowledge base for M.Tech in Artificial Intelligence and Machine Learning. This documentation serves as a centralized repository for course notes, concepts, implementations, and resources.

## 📚 Contents

- **Semester 1**
  - Mathematical Foundations for Machine Learning
  - Deep Neural Networks
  - Introduction to Statistical Methods
  - Machine Learning

*More semesters to be added as the program progresses.*

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shivam2003-dev/wilp_mtech-aiml-knowledge-base.git
   cd wilp_mtech-aiml-knowledge-base
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Serve locally**
   ```bash
   mkdocs serve
   ```

4. **Open in browser**
   Navigate to `http://127.0.0.1:8000/` to view the documentation.

## 🏗️ Building the Documentation

To build the static site:

```bash
mkdocs build
```

The built site will be in the `site/` directory.

## 📖 Structure

```
mtech-aiml-knowledge-base/
├── docs/                      # Documentation source files
│   ├── index.md              # Homepage
│   ├── semester1/            # Semester 1 content
│   │   ├── index.md
│   │   ├── mathematical-foundations.md
│   │   ├── deep-neural-networks.md
│   │   ├── statistical-methods.md
│   │   └── machine-learning.md
│   ├── about/                # About section
│   │   ├── program.md
│   │   └── contact.md
│   ├── stylesheets/          # Custom CSS
│   │   └── extra.css
│   └── javascripts/          # Custom JavaScript
│       └── mathjax.js
├── mkdocs.yml                # MkDocs configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## ✨ Features

- 🎨 **Beautiful Material Design** - Clean and modern interface
- 🔍 **Powerful Search** - Find topics quickly
- 🌓 **Dark/Light Mode** - Toggle between themes
- 📱 **Responsive** - Works on all devices
- 🧮 **Math Support** - LaTeX equations with MathJax
- 💻 **Code Highlighting** - Syntax highlighting for multiple languages
- 📊 **Diagrams** - Support for Mermaid diagrams
- 🔗 **Easy Navigation** - Organized by semester and subject

The site will be available at: `https://shivam2003-dev.github.io/wilp_mtech-aiml-knowledge-base/`

## 🛠️ Technology Stack

- **[MkDocs](https://www.mkdocs.org/)** - Static site generator
- **[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)** - Theme
- **[Python Markdown Extensions](https://facelessuser.github.io/pymdown-extensions/)** - Enhanced markdown
- **[MathJax](https://www.mathjax.org/)** - Mathematical equations

## 📝 Contributing

Contributions are welcome! Whether you want to fix a typo, add notes, upload study materials, or improve the documentation, your help is appreciated.

### How to Contribute

#### 1. **Fork and Clone**
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/wilp_mtech-aiml-knowledge-base.git
cd wilp_mtech-aiml-knowledge-base

# Add upstream remote
git remote add upstream https://github.com/shivam2003-dev/wilp_mtech-aiml-knowledge-base.git
```

#### 2. **Create a Branch**
```bash
# Create a new branch for your contribution
git checkout -b feature/add-notes
# or
git checkout -b fix/typo-correction
```

#### 3. **Make Your Changes**

**Adding Markdown Content:**
- Edit existing `.md` files in the `docs/` directory
- Follow the existing formatting and structure
- Use proper heading levels (### for main sections)

**Adding PDFs or Files:**

1. Create a `files` or `resources` directory in the appropriate semester folder:
   ```bash
   mkdir -p docs/semester1/files
   ```

2. Add your PDF files:
   ```bash
   cp /path/to/your/file.pdf docs/semester1/files/
   ```

3. Link to the PDF in your markdown file:
   ```markdown
   ## 📄 Study Materials
   
   - [Lecture Notes - Week 1](files/lecture-week1.pdf)
   - [Assignment Solutions](files/assignment-solutions.pdf)
   - [Reference Book Chapter 3](files/chapter3.pdf)
   ```

**Adding External Links:**
```markdown
## 🔗 Useful Resources

- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Andrew Ng's ML Course](https://www.coursera.org/learn/machine-learning)
- [Fast.ai Courses](https://www.fast.ai/)
```

**Adding Code Examples:**
````markdown
## � Code Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Your code here
def example_function():
    return "Hello, World!"
```
````

#### 4. **Test Locally**
```bash
# Install dependencies
pip install -r requirements.txt

# Serve locally to preview changes
mkdocs serve

# Open http://127.0.0.1:8000 in your browser
```

#### 5. **Commit Your Changes**
```bash
git add .
git commit -m "Add: lecture notes for Deep Neural Networks week 1"
# or
git commit -m "Fix: typo in mathematical foundations"
# or
git commit -m "Update: add reference links for ML algorithms"
```

**Commit Message Guidelines:**
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Prefix with type: `Add:`, `Fix:`, `Update:`, `Remove:`, `Docs:`

#### 6. **Push to Your Fork**
```bash
git push origin feature/add-notes
```

#### 7. **Create a Pull Request**

1. Go to your fork on GitHub
2. Click "Compare & Pull Request" button
3. Fill in the PR template:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New content (notes, resources)
   - [ ] Documentation update
   - [ ] Code examples
   
   ## Checklist
   - [ ] Tested locally with `mkdocs serve`
   - [ ] Follows existing formatting
   - [ ] No broken links
   - [ ] Proper file structure
   ```
4. Submit the pull request

### 📋 Contribution Guidelines

**Content Quality:**
- ✅ Clear and concise explanations
- ✅ Proper grammar and spelling
- ✅ Well-formatted code with comments
- ✅ Accurate information with sources
- ✅ Follows existing structure

**File Organization:**
- ✅ Place files in appropriate semester/subject folders
- ✅ Use descriptive filenames (lowercase with hyphens)
- ✅ Keep images/PDFs in `files/` subdirectories
- ✅ Update navigation in `mkdocs.yml` if adding new pages

**What to Contribute:**
- 📝 Lecture notes and summaries
- 📄 Assignment solutions (with permission)
- 🔗 Useful resources and links
- 💻 Code implementations
- 📊 Diagrams and visualizations
- 🐛 Bug fixes and corrections
- 📖 Additional explanations

**Before Submitting:**
- [ ] Test with `mkdocs serve`
- [ ] Check for broken links
- [ ] Ensure PDFs are properly linked
- [ ] Verify math equations render correctly
- [ ] Check code syntax highlighting works

### 🤝 Types of Contributions Welcome

1. **Content Additions**: Add notes, examples, explanations
2. **Resource Links**: Share useful learning materials
3. **Code Examples**: Provide implementations of algorithms
4. **Bug Fixes**: Fix typos, broken links, formatting issues
5. **Documentation**: Improve README, add comments
6. **Suggestions**: Open issues for new features or improvements

### 📞 Questions?

If you have questions about contributing:
- 💬 Open an issue for discussion
- 📧 Connect via [LinkedIn](https://www.linkedin.com/in/shivam-kumar2003/)
- 📝 Check existing issues and PRs for examples

## 🌐 Deployment

### GitHub Pages

1. **Configure GitHub Pages**
   - Go to repository Settings → Pages
   - Select source as GitHub Actions

2. **Deploy using GitHub Actions**
   The site will automatically deploy when changes are pushed to the main branch.

   Alternatively, deploy manually:
   ```bash
   mkdocs gh-deploy
   ```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Shivam Kumar**

- 💼 LinkedIn: [@shivam-kumar2003](https://www.linkedin.com/in/shivam-kumar2003/)
- 🐙 GitHub: [@shivam2003-dev](https://github.com/shivam2003-dev)

## 🙏 Acknowledgments

- Thanks to all the professors and instructors
- The open-source community for the amazing tools
- Fellow students for collaborative learning

## 📊 Project Status

![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)
![Semester](https://img.shields.io/badge/Semester-1-blue?style=flat-square)
![Last Updated](https://img.shields.io/badge/Last%20Updated-October%202025-orange?style=flat-square)

## 🗺️ Roadmap

- [x] Semester 1 content
- [ ] Semester 2 content (Coming soon)
- [ ] Add video tutorials
- [ ] Interactive code examples
- [ ] Practice problems and solutions
- [ ] Research paper summaries
- [ ] Project showcases

## 📞 Support

If you have any questions or need help:

- 📧 Open an issue on GitHub
- 💬 Connect via LinkedIn
- 🐛 Report bugs in the Issues section

---

<div align="center">
  <p><strong>Made with ❤️ and MkDocs Material</strong></p>
  <p><em>Happy Learning! 🚀</em></p>
</div>
