import React from "react";
import Title from "../../../components/Title/Title";

const ReviewerInstruction2023page = () => {
  return (
    <div>
      <Title>Reviewer Instructions</Title>
      <p>Thank you for agreeing to review CLearR 2023! Your assessment is vital to creating a high quality program. This page provides the review guidelines that will help you to write reviews efficiently and effectively.</p>
      <h5>Main tasks</h5>
      <ol>
      <li><b>Preparation (by Oct 28, 2022)</b></li>
        <ul>
          <li>CLeaR 2023 is using the OpenReview System. Please create your OpenReview profile if you do not have one and make sure it is up to date if you already have an account. </li>
          <li>Reviewer invitations will be sent via noreply@openreview.net. Please accept the reviewer invitation before the expiry date.</li>
          <li>Please read and agree to CleaR 2023 <a href="CodeofConduct">codes of conduct</a> and declare the right <a href="ConflictsofInterest">conflicts of interests</a>.</li>
        </ul>
         <li><b>Paper bidding and assignments checking (Oct 31, 2022 - Nov 4, 2022)</b></li>
        <ul>
          <li>Please bid on the papers that fall into your area of expertise. Your bidding is an important input to the overall matching results. </li>
          <li>Please check the assigned papers right after the paper assignment. If you do not feel qualified to review a paper or find potential conflicts of interest, please communicate with your AC as soon as possible.</li>
        </ul>
        <li><b>Write thorough and timely reviews: (Nov 10, 2022 - Nov 29, 2022)</b></li>
        <ul>
          <li>Please make your review as deep and detailed as possible. Superficial reviews are not really helpful in making final decisions. It is also important to treat each submission fairly and provide unbiased reviews. </li>
          <li>A review form has been designed to facilitate the review process. Please refer to the “Review Form” section for a step-by-step instruction on how to answer each question in the review form.</li>
        </ul>
        <li><b>Discuss with authors/fellow reviewers/ACs (Dec 12, 2022 -- Dec 30, 2022)</b></li>
        <ul>
          <li>Before the start of discussions, please carefully read author responses with an open mind to avoid possible misunderstandings. Even if the author's rebuttal does not change your opinion, please acknowledge that you have read and considered it. </li>
          <li>A further discussion with the authors will be enabled during the discussion period.  If you want the authors to clarify more things after reading the rebuttal, you can discuss with them on the paper’s page.</li>
          <li>All reviewers should actively participate in discussions with fellow reviewers and ACs to have a more comprehensive understanding of each paper. The discussions are especially important for borderline papers and papers with high variance assessments. While engaging in the discussion, please be professional, polite, and keep an open mind. Although full consensus makes the final decision easier, it is not mandatory in the reviewing process, as different people may have different perspectives.</li>
          <li>If you change your opinion during or after the discussion phase,  please update your ratings and give specific reasons in the final comments. </li>
        </ul>
      </ol>
      <h5>Review form</h5>
      <ol>
        <li><b>Summary</b>. Summarize the main contributions of each paper. The contributions may be new problems, theories, methods, algorithms, applications, benchmarks, etc. </li>
        <li><b>Main review</b>. Please provide an in-depth review of each paper by considering the following aspects: </li>
        <ul>
          <li>Originality: Does the paper provide anything new, like a new problem or a new method? Is the novelty compared to existing works well justified? Is it possible that similar ideas have been studied but the paper does not cite them properly? </li>
          <li>Significance: Does the paper address an important problem? How relevant are the results to the CLeaR community? Does the proposed theory or method significantly advance the state-of-the-art? Do the results in the paper provide new insights to the research problem? Is this paper likely to have broad impacts outside the CLeaR community, e.g., in natural/social science or engineering?</li>
          <li>Technical quality: Is the proposed approach technically sound? Are claims substantiated by theoretical and/or empirical results? Are the derivations and proofs correct? Is the proposed method unnecessarily complicated? Are the hyperparameters tuned in an appropriate manner?</li>
          <li>Clarity: Is the submission clearly written and well organized? Is the take home message easily extractable from the paper? Is the motivation well explained by illustrations and examples? Are the technical details described rigorously? Is there a significant amount of typos that make the paper hard to read?</li>
        </ul>
        <li><b>Overall score</b>. We use a 10-point scoring system for the overall assessment. Please select the category that best describes your assessment of the paper.</li>
        <ul>
          <li>10: Top 5% of accepted papers, seminal paper</li>
          <li>9: Top 15% of accepted papers, strong accept</li>
          <li>8: Top 50% of accepted papers, clear accept</li>
          <li>7: Good paper, accept</li>
          <li>6: Marginally above acceptance threshold</li>
          <li>5: Marginally below acceptance threshold</li>
          <li>4: Ok but not good enough - rejection</li>
          <li>3: Clear rejection</li>
          <li>2: Strong rejection</li>
          <li>1: Trivial or wrong</li>
        </ul>
      <li><b>Confidence score</b>. Please select the category that best describes your confidence in the assessment of the submission.</li>
      <ul>
          <li>5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.</li>
          <li>4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.</li>
          <li>3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.</li>
          <li>2: You are willing to defend your assessment, but it is quite likely that you did not understand central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.</li>
          <li>1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.</li>
        </ul>
      </ol>
      <h5> Policies</h5>
      <p><b>Confidentiality.</b> By reviewing CleaR 2023, you must agree to keep all material and information related to the review confidential. In particular, you must not use ideas and results from submitted papers in your own research or distribute them to others. You should delete all reviewing material, such as the submitted code, at the end of the reviewing cycle. You should not talk about submissions or content related to the reviewing of submissions to anyone without prior approval from the program chairs.</p>
      <p><b>Double-blind reviewing.</b> The CLeaR review process is double-blind: reviewers and authors will both stay anonymous to each other during the review process. However, author names will be visible to area chairs and program chairs. Authors are responsible for anonymizing their submissions. Submissions may not contain any identifying information that may violate the double-blind reviewing policy.  If you are assigned a submission that is not adequately anonymized, please contact the corresponding AC. Also, you should not attempt to find out the identities of authors for any of your assigned submissions, e.g., by searching arXiv preprints. Reviewer names are visible to the area chair (and program chairs), but the reviewers will not know names of other reviewers. Please do not disclose your identity to authors and fellow reviewers in the discussions.</p>
      <p><b>Dual Submissions.</b>CLeaR does not allow double submissions. Namely, submissions should not have been previously published in or submitted to a journal or the proceedings of another conference at any point during the CLeaR review process. Submissions as extended abstracts (5 pages or less), to workshops or non-archival venues (without a proceedings), will not be considered a concurrent submission. Authors may submit anonymized work to CLeaR that is already available as a preprint (e.g., on arXiv) without citing it. If you suspect that a submission that has been assigned to you is a dual submission or if you require further clarification, please contact the corresponding AC. Please see Call for Papers for more information about dual submissions.</p>
      <p><b>Violations of formatting instructions.</b> Submissions are limited to 12 single-column PMLR-formatted pages, plus unlimited additional pages for references and appendices. Authors of accepted papers will have the option of opting out of the proceedings in favor of a 1-page extended abstract, which will point to an open access archival version of the full paper reviewed for CLeaR. If you are assigned a paper that is overlength or appears to violate the CLeaR proceedings format (e.g., by decreasing margins or font size, by removing some pre-fixed spaces, etc), please notify the corresponding AC immediately.</p>
      
      <p> * <b>Please also review the policies in the CLeaR 2023 <a href="CallforPapers">Call for Papers</a>.</b> </p>
    </div>
  )
}

export default ReviewerInstruction2023page
