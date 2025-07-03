import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

interface D3PieChartProps {
  data: { label: string; value: number; color?: string }[];
  width?: number;
  height?: number;
  title?: string;
  style?: React.CSSProperties;
}

export const D3PieChart: React.FC<D3PieChartProps> = ({
  data,
  width = 320,
  height = 320,
  title,
  style = {},
}) => {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!data || data.length === 0) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const radius = Math.min(width, height) / 2 - 40;
    const innerRadius = radius * 0.4; // Donut chart

    // Arc generator
    const arc = d3
      .arc<d3.PieArcDatum<{ label: string; value: number; color?: string }>>()
      .innerRadius(innerRadius)
      .outerRadius(radius);

    // Pie generator
    const pie = d3
      .pie<{ label: string; value: number; color?: string }>()
      .value((d) => d.value)
      .sort(null);

    // Color scale
    const color = d3.scaleOrdinal(d3.schemeCategory10);

    // Tooltip
    const tooltip = d3
      .select(ref.current?.parentElement)
      .selectAll('.d3-tooltip')
      .data([null])
      .join('div')
      .attr('class', 'd3-tooltip')
      .style('position', 'absolute')
      .style('pointer-events', 'none')
      .style('background', 'rgba(255,255,255,0.95)')
      .style('border', '1px solid #bbb')
      .style('border-radius', '4px')
      .style('padding', '6px 10px')
      .style('font-size', '13px')
      .style('color', '#222')
      .style('box-shadow', '0 2px 8px rgba(0,0,0,0.07)')
      .style('display', 'none');

    // Main group
    const g = svg
      .append('g')
      .attr('transform', `translate(${width / 2},${height / 2})`);

    // Pie slices
    const arcs = g
      .selectAll('.arc')
      .data(pie(data))
      .join('g')
      .attr('class', 'arc');

    arcs
      .append('path')
      .attr('d', arc)
      .attr('fill', (d, i) => d.data.color || color(i.toString()))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('opacity', 0.8)
      .on('mouseover', function (event, d) {
        d3.select(this).style('opacity', 1);
        const percentage = ((d.data.value / d3.sum(data, (d) => d.value)) * 100).toFixed(1);
        tooltip
          .style('display', 'block')
          .html(`<strong>${d.data.label}</strong><br/>Value: ${d.data.value.toLocaleString()}<br/>Percentage: ${percentage}%`)
          .style('left', event.pageX + 10 + 'px')
          .style('top', event.pageY - 28 + 'px');
      })
      .on('mouseout', function () {
        d3.select(this).style('opacity', 0.8);
        tooltip.style('display', 'none');
      });

    // Labels
    arcs
      .append('text')
      .attr('transform', (d) => `translate(${arc.centroid(d)})`)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('fill', '#333')
      .attr('font-weight', 'bold')
      .text((d) => {
        const percentage = ((d.data.value / d3.sum(data, (d) => d.value)) * 100);
        return percentage > 5 ? `${percentage.toFixed(0)}%` : '';
      });

    // Title
    if (title) {
      svg
        .append('text')
        .attr('x', width / 2)
        .attr('y', 25)
        .attr('text-anchor', 'middle')
        .attr('font-size', 18)
        .attr('font-family', 'Georgia, serif')
        .attr('font-weight', 'bold')
        .attr('fill', '#222')
        .text(title);
    }

    // Legend
    const legend = svg
      .append('g')
      .attr('transform', `translate(${width - 120}, 40)`);

    const legendItems = legend
      .selectAll('.legend-item')
      .data(data)
      .join('g')
      .attr('class', 'legend-item')
      .attr('transform', (d, i) => `translate(0, ${i * 20})`);

    legendItems
      .append('rect')
      .attr('width', 12)
      .attr('height', 12)
      .attr('fill', (d, i) => d.color || color(i.toString()));

    legendItems
      .append('text')
      .attr('x', 18)
      .attr('y', 10)
      .attr('font-size', 12)
      .attr('fill', '#333')
      .text((d) => d.label);
  }, [data, width, height, title]);

  return (
    <div style={{ position: 'relative', width, height, ...style }}>
      <svg ref={ref} width={width} height={height} />
    </div>
  );
}; 