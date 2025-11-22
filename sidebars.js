/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Semester 1',
      items: [
        'semester1/index',
        {
          type: 'category',
          label: 'Courses',
          items: [
            'semester1/mathematical-foundations',
            'semester1/deep-neural-networks',
            'semester1/statistical-methods',
            'semester1/machine-learning',
          ],
        },
        {
          type: 'category',
          label: 'Assignments',
          items: [
            {
              type: 'category',
              label: 'Statistical Methods',
              items: [
                {
                  type: 'category',
                  label: 'Assignment 1',
                  items: [
                    'semester1/assignments/statistical-methods/assignment1/index',
                  ],
                },
              ],
            },
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'About',
      items: [
        'about/program',
        'about/contact',
      ],
    },
  ],
};

module.exports = sidebars;

