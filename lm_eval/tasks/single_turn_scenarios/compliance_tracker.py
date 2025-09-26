#!/usr/bin/env python3
"""
Compliance tracking and audit trail functionality for single_turn_scenarios.
Implements licensing compliance tracking, audit procedures, and third-party code management.

Requirements: 12.2, 12.3
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class LicenseInfo:
    """Information about a license."""
    spdx_id: str
    name: str
    url: str
    is_compatible: bool
    requires_attribution: bool
    requires_source_disclosure: bool
    commercial_use_allowed: bool

@dataclass
class ComplianceRecord:
    """Record of compliance check for a component."""
    component_id: str
    component_type: str  # 'problem', 'test', 'reference', 'third_party'
    license: str
    author: str
    source_url: Optional[str]
    attribution_text: Optional[str]
    checksum: str
    timestamp: str
    compliance_status: str  # 'compliant', 'non_compliant', 'needs_review'
    issues: List[str]
    reviewer: Optional[str] = None
    review_date: Optional[str] = None

@dataclass
class AuditTrail:
    """Audit trail for compliance tracking."""
    audit_id: str
    timestamp: str
    auditor: str
    scope: str  # 'full', 'incremental', 'targeted'
    total_components: int
    compliant_components: int
    non_compliant_components: int
    needs_review_components: int
    issues_found: List[str]
    recommendations: List[str]
    next_audit_date: Optional[str] = None

class ComplianceTracker:
    """Tracks licensing compliance and maintains audit trails."""
    
    # Known compatible licenses with MIT
    COMPATIBLE_LICENSES = {
        'MIT': LicenseInfo(
            spdx_id='MIT',
            name='MIT License',
            url='https://opensource.org/licenses/MIT',
            is_compatible=True,
            requires_attribution=True,
            requires_source_disclosure=False,
            commercial_use_allowed=True
        ),
        'Apache-2.0': LicenseInfo(
            spdx_id='Apache-2.0',
            name='Apache License 2.0',
            url='https://opensource.org/licenses/Apache-2.0',
            is_compatible=True,
            requires_attribution=True,
            requires_source_disclosure=False,
            commercial_use_allowed=True
        ),
        'BSD-3-Clause': LicenseInfo(
            spdx_id='BSD-3-Clause',
            name='BSD 3-Clause License',
            url='https://opensource.org/licenses/BSD-3-Clause',
            is_compatible=True,
            requires_attribution=True,
            requires_source_disclosure=False,
            commercial_use_allowed=True
        ),
        'CC0-1.0': LicenseInfo(
            spdx_id='CC0-1.0',
            name='Creative Commons Zero v1.0 Universal',
            url='https://creativecommons.org/publicdomain/zero/1.0/',
            is_compatible=True,
            requires_attribution=False,
            requires_source_disclosure=False,
            commercial_use_allowed=True
        ),
        'Unlicense': LicenseInfo(
            spdx_id='Unlicense',
            name='The Unlicense',
            url='https://unlicense.org/',
            is_compatible=True,
            requires_attribution=False,
            requires_source_disclosure=False,
            commercial_use_allowed=True
        )
    }
    
    # Incompatible licenses
    INCOMPATIBLE_LICENSES = {
        'GPL-3.0': LicenseInfo(
            spdx_id='GPL-3.0',
            name='GNU General Public License v3.0',
            url='https://opensource.org/licenses/GPL-3.0',
            is_compatible=False,
            requires_attribution=True,
            requires_source_disclosure=True,
            commercial_use_allowed=True
        ),
        'AGPL-3.0': LicenseInfo(
            spdx_id='AGPL-3.0',
            name='GNU Affero General Public License v3.0',
            url='https://opensource.org/licenses/AGPL-3.0',
            is_compatible=False,
            requires_attribution=True,
            requires_source_disclosure=True,
            commercial_use_allowed=True
        )
    }
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize compliance tracker.
        
        Args:
            base_dir: Base directory for the task (defaults to current file's parent).
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent
        else:
            self.base_dir = Path(base_dir)
        
        self.compliance_dir = self.base_dir / "compliance"
        self.compliance_dir.mkdir(exist_ok=True)
        
        self.records_file = self.compliance_dir / "compliance_records.json"
        self.audit_file = self.compliance_dir / "audit_trail.json"
        self.third_party_file = self.compliance_dir / "third_party_licenses.json"
        
        self._load_existing_records()
    
    def _load_existing_records(self):
        """Load existing compliance records."""
        self.compliance_records: Dict[str, ComplianceRecord] = {}
        self.audit_trails: List[AuditTrail] = []
        self.third_party_licenses: Dict[str, Dict] = {}
        
        # Load compliance records
        if self.records_file.exists():
            try:
                with open(self.records_file, 'r') as f:
                    data = json.load(f)
                    for record_data in data:
                        record = ComplianceRecord(**record_data)
                        self.compliance_records[record.component_id] = record
            except Exception as e:
                logger.warning(f"Failed to load compliance records: {e}")
        
        # Load audit trails
        if self.audit_file.exists():
            try:
                with open(self.audit_file, 'r') as f:
                    data = json.load(f)
                    self.audit_trails = [AuditTrail(**trail_data) for trail_data in data]
            except Exception as e:
                logger.warning(f"Failed to load audit trails: {e}")
        
        # Load third-party licenses
        if self.third_party_file.exists():
            try:
                with open(self.third_party_file, 'r') as f:
                    self.third_party_licenses = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load third-party licenses: {e}")
    
    def _save_records(self):
        """Save compliance records to file."""
        try:
            records_data = [asdict(record) for record in self.compliance_records.values()]
            with open(self.records_file, 'w') as f:
                json.dump(records_data, f, indent=2)
            
            audit_data = [asdict(trail) for trail in self.audit_trails]
            with open(self.audit_file, 'w') as f:
                json.dump(audit_data, f, indent=2)
            
            with open(self.third_party_file, 'w') as f:
                json.dump(self.third_party_licenses, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save compliance records: {e}")
    
    def _calculate_checksum(self, content: str) -> str:
        """Calculate SHA-256 checksum of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()
    
    def validate_license_compatibility(self, license_id: str) -> Tuple[bool, List[str]]:
        """Validate if a license is compatible with the project.
        
        Args:
            license_id: SPDX license identifier.
        
        Returns:
            Tuple of (is_compatible, issues).
        """
        issues = []
        
        if license_id in self.COMPATIBLE_LICENSES:
            return True, issues
        
        if license_id in self.INCOMPATIBLE_LICENSES:
            license_info = self.INCOMPATIBLE_LICENSES[license_id]
            issues.append(f"License {license_id} is incompatible with MIT license")
            if license_info.requires_source_disclosure:
                issues.append(f"License {license_id} requires source code disclosure")
            return False, issues
        
        # Unknown license
        issues.append(f"Unknown license {license_id} - manual review required")
        return False, issues
    
    def register_component(self, 
                          component_id: str,
                          component_type: str,
                          license_id: str,
                          author: str,
                          content: str,
                          source_url: Optional[str] = None,
                          attribution_text: Optional[str] = None) -> ComplianceRecord:
        """Register a component for compliance tracking.
        
        Args:
            component_id: Unique identifier for the component.
            component_type: Type of component ('problem', 'test', 'reference', 'third_party').
            license_id: SPDX license identifier.
            author: Author or copyright holder.
            content: Content of the component for checksum calculation.
            source_url: URL of the original source (if applicable).
            attribution_text: Required attribution text (if applicable).
        
        Returns:
            ComplianceRecord for the registered component.
        """
        # Validate license compatibility
        is_compatible, issues = self.validate_license_compatibility(license_id)
        
        # Determine compliance status
        if is_compatible and not issues:
            compliance_status = 'compliant'
        elif issues:
            compliance_status = 'needs_review' if is_compatible else 'non_compliant'
        else:
            compliance_status = 'compliant'
        
        # Create compliance record
        record = ComplianceRecord(
            component_id=component_id,
            component_type=component_type,
            license=license_id,
            author=author,
            source_url=source_url,
            attribution_text=attribution_text,
            checksum=self._calculate_checksum(content),
            timestamp=self._get_current_timestamp(),
            compliance_status=compliance_status,
            issues=issues
        )
        
        # Store record
        self.compliance_records[component_id] = record
        self._save_records()
        
        logger.info(f"Registered component {component_id} with status {compliance_status}")
        return record
    
    def update_component_status(self, 
                               component_id: str,
                               new_status: str,
                               reviewer: str,
                               notes: Optional[str] = None):
        """Update the compliance status of a component.
        
        Args:
            component_id: Component identifier.
            new_status: New compliance status.
            reviewer: Person performing the review.
            notes: Additional notes about the status change.
        """
        if component_id not in self.compliance_records:
            raise ValueError(f"Component {component_id} not found in compliance records")
        
        record = self.compliance_records[component_id]
        record.compliance_status = new_status
        record.reviewer = reviewer
        record.review_date = self._get_current_timestamp()
        
        if notes:
            record.issues.append(f"Review note: {notes}")
        
        self._save_records()
        logger.info(f"Updated component {component_id} status to {new_status}")
    
    def register_third_party_license(self, 
                                   library_name: str,
                                   license_id: str,
                                   license_text: str,
                                   source_url: str,
                                   version: Optional[str] = None):
        """Register a third-party library license.
        
        Args:
            library_name: Name of the third-party library.
            license_id: SPDX license identifier.
            license_text: Full text of the license.
            source_url: URL where the license was obtained.
            version: Version of the library (if applicable).
        """
        license_record = {
            'library_name': library_name,
            'license_id': license_id,
            'license_text': license_text,
            'source_url': source_url,
            'version': version,
            'registered_date': self._get_current_timestamp(),
            'checksum': self._calculate_checksum(license_text)
        }
        
        self.third_party_licenses[library_name] = license_record
        self._save_records()
        
        logger.info(f"Registered third-party license for {library_name}")
    
    def perform_compliance_audit(self, 
                                auditor: str,
                                scope: str = 'full') -> AuditTrail:
        """Perform a compliance audit.
        
        Args:
            auditor: Person performing the audit.
            scope: Scope of the audit ('full', 'incremental', 'targeted').
        
        Returns:
            AuditTrail record of the audit.
        """
        audit_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Count compliance statuses
        status_counts = defaultdict(int)
        for record in self.compliance_records.values():
            status_counts[record.compliance_status] += 1
        
        # Identify issues
        issues_found = []
        recommendations = []
        
        # Check for non-compliant components
        non_compliant = [r for r in self.compliance_records.values() 
                        if r.compliance_status == 'non_compliant']
        if non_compliant:
            issues_found.append(f"Found {len(non_compliant)} non-compliant components")
            recommendations.append("Review and resolve non-compliant components")
        
        # Check for components needing review
        needs_review = [r for r in self.compliance_records.values() 
                       if r.compliance_status == 'needs_review']
        if needs_review:
            issues_found.append(f"Found {len(needs_review)} components needing review")
            recommendations.append("Complete manual review of flagged components")
        
        # Check for missing attribution
        missing_attribution = [r for r in self.compliance_records.values() 
                              if r.license in self.COMPATIBLE_LICENSES 
                              and self.COMPATIBLE_LICENSES[r.license].requires_attribution
                              and not r.attribution_text]
        if missing_attribution:
            issues_found.append(f"Found {len(missing_attribution)} components missing required attribution")
            recommendations.append("Add attribution text for components requiring it")
        
        # Check for outdated records (older than 6 months)
        six_months_ago = datetime.now(timezone.utc).timestamp() - (6 * 30 * 24 * 3600)
        outdated_records = [r for r in self.compliance_records.values()
                           if datetime.fromisoformat(r.timestamp.replace('Z', '+00:00')).timestamp() < six_months_ago]
        if outdated_records:
            issues_found.append(f"Found {len(outdated_records)} records older than 6 months")
            recommendations.append("Review and update outdated compliance records")
        
        # Create audit trail
        audit_trail = AuditTrail(
            audit_id=audit_id,
            timestamp=self._get_current_timestamp(),
            auditor=auditor,
            scope=scope,
            total_components=len(self.compliance_records),
            compliant_components=status_counts['compliant'],
            non_compliant_components=status_counts['non_compliant'],
            needs_review_components=status_counts['needs_review'],
            issues_found=issues_found,
            recommendations=recommendations,
            next_audit_date=datetime.now().replace(month=datetime.now().month + 3).isoformat()
        )
        
        self.audit_trails.append(audit_trail)
        self._save_records()
        
        logger.info(f"Completed compliance audit {audit_id}")
        return audit_trail
    
    def generate_compliance_report(self) -> str:
        """Generate a comprehensive compliance report.
        
        Returns:
            Formatted compliance report as string.
        """
        report = f"""# Compliance Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Components**: {len(self.compliance_records)}
- **Compliant**: {sum(1 for r in self.compliance_records.values() if r.compliance_status == 'compliant')}
- **Non-Compliant**: {sum(1 for r in self.compliance_records.values() if r.compliance_status == 'non_compliant')}
- **Needs Review**: {sum(1 for r in self.compliance_records.values() if r.compliance_status == 'needs_review')}

## License Distribution

"""
        
        # License distribution
        license_counts = defaultdict(int)
        for record in self.compliance_records.values():
            license_counts[record.license] += 1
        
        for license_id, count in sorted(license_counts.items()):
            compatibility = "✅ Compatible" if license_id in self.COMPATIBLE_LICENSES else "❌ Incompatible"
            report += f"- **{license_id}**: {count} components ({compatibility})\n"
        
        # Component type distribution
        report += "\n## Component Types\n\n"
        type_counts = defaultdict(int)
        for record in self.compliance_records.values():
            type_counts[record.component_type] += 1
        
        for comp_type, count in sorted(type_counts.items()):
            report += f"- **{comp_type.replace('_', ' ').title()}**: {count} components\n"
        
        # Non-compliant components
        non_compliant = [r for r in self.compliance_records.values() 
                        if r.compliance_status == 'non_compliant']
        if non_compliant:
            report += "\n## Non-Compliant Components\n\n"
            for record in non_compliant:
                report += f"- **{record.component_id}** ({record.component_type})\n"
                report += f"  - License: {record.license}\n"
                report += f"  - Issues: {', '.join(record.issues)}\n"
        
        # Components needing review
        needs_review = [r for r in self.compliance_records.values() 
                       if r.compliance_status == 'needs_review']
        if needs_review:
            report += "\n## Components Needing Review\n\n"
            for record in needs_review:
                report += f"- **{record.component_id}** ({record.component_type})\n"
                report += f"  - License: {record.license}\n"
                report += f"  - Issues: {', '.join(record.issues)}\n"
        
        # Third-party licenses
        if self.third_party_licenses:
            report += "\n## Third-Party Licenses\n\n"
            for lib_name, lib_info in self.third_party_licenses.items():
                report += f"- **{lib_name}** ({lib_info.get('version', 'unknown version')})\n"
                report += f"  - License: {lib_info['license_id']}\n"
                report += f"  - Source: {lib_info['source_url']}\n"
        
        # Recent audits
        if self.audit_trails:
            report += "\n## Recent Audits\n\n"
            recent_audits = sorted(self.audit_trails, key=lambda x: x.timestamp, reverse=True)[:5]
            for audit in recent_audits:
                report += f"- **{audit.audit_id}** ({audit.timestamp[:10]})\n"
                report += f"  - Auditor: {audit.auditor}\n"
                report += f"  - Scope: {audit.scope}\n"
                report += f"  - Issues Found: {len(audit.issues_found)}\n"
        
        return report
    
    def export_attribution_file(self) -> str:
        """Export attribution file for components requiring attribution.
        
        Returns:
            Attribution text for all components requiring it.
        """
        attribution_text = "# Attribution\n\n"
        attribution_text += "This project includes components from the following sources:\n\n"
        
        for record in self.compliance_records.values():
            if (record.license in self.COMPATIBLE_LICENSES and 
                self.COMPATIBLE_LICENSES[record.license].requires_attribution):
                
                attribution_text += f"## {record.component_id}\n\n"
                attribution_text += f"- **Author**: {record.author}\n"
                attribution_text += f"- **License**: {record.license}\n"
                
                if record.source_url:
                    attribution_text += f"- **Source**: {record.source_url}\n"
                
                if record.attribution_text:
                    attribution_text += f"- **Attribution**: {record.attribution_text}\n"
                
                attribution_text += "\n"
        
        # Add third-party licenses
        for lib_name, lib_info in self.third_party_licenses.items():
            attribution_text += f"## {lib_name}\n\n"
            attribution_text += f"- **License**: {lib_info['license_id']}\n"
            attribution_text += f"- **Source**: {lib_info['source_url']}\n"
            if lib_info.get('version'):
                attribution_text += f"- **Version**: {lib_info['version']}\n"
            attribution_text += "\n"
        
        return attribution_text

def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compliance tracking and audit tool')
    parser.add_argument('--audit', action='store_true', help='Perform compliance audit')
    parser.add_argument('--auditor', default='system', help='Name of auditor')
    parser.add_argument('--report', action='store_true', help='Generate compliance report')
    parser.add_argument('--attribution', action='store_true', help='Export attribution file')
    parser.add_argument('--output', help='Output file for reports')
    
    args = parser.parse_args()
    
    tracker = ComplianceTracker()
    
    if args.audit:
        audit_trail = tracker.perform_compliance_audit(args.auditor)
        print(f"Audit completed: {audit_trail.audit_id}")
        print(f"Issues found: {len(audit_trail.issues_found)}")
        print(f"Recommendations: {len(audit_trail.recommendations)}")
    
    if args.report:
        report = tracker.generate_compliance_report()
        if args.output:
            Path(args.output).write_text(report)
            print(f"Compliance report saved to {args.output}")
        else:
            print(report)
    
    if args.attribution:
        attribution = tracker.export_attribution_file()
        output_file = args.output or "ATTRIBUTION.md"
        Path(output_file).write_text(attribution)
        print(f"Attribution file saved to {output_file}")

if __name__ == '__main__':
    main()