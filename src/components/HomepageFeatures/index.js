import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Semester 1',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Mathematical Foundations, Deep Neural Networks, Statistical Methods, and Machine Learning.
        Comprehensive course materials, notes, and assignments.
      </>
    ),
    link: '/docs/semester1',
  },
  {
    title: 'Assignments & Projects',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Complete solutions with step-by-step implementations, mathematical derivations, 
        and visualizations for all assignments.
      </>
    ),
    link: '/docs/semester1/assignments/statistical-methods/assignment1',
  },
  {
    title: 'Resources & Materials',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Textbooks, research papers, video lectures, and additional learning resources 
        recommended by professors and the community.
      </>
    ),
    link: '/docs/semester1',
  },
];

function Feature({Svg, title, description, link}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
        <a href={link} className="button button--secondary">Explore →</a>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

